#!/usr/bin/env python3
"""
factor_lab_tui.py

A Textual-based Mission Control interface for Factor Lab.
Run this file to launch the GUI.
"""

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.widgets import (
    Button, Header, Footer, Static, Label, Input, 
    Switch, DataTable, RichLog, TabbedContent, TabPane, Select
)
from textual import on, work
from loguru import logger
import sys

# Import Adapter
from factor_lab_adapter import FactorLabAdapter

# ==============================================================================
# STYLING (CSS)
# ==============================================================================
APP_CSS = """
Screen {
    layout: horizontal;
}

/* Sidebar */
#sidebar {
    dock: left;
    width: 30;
    height: 100%;
    background: $panel;
    border-right: vkey $primary;
}

#sidebar Label {
    padding: 1;
    text-align: center;
    text-style: bold;
}

#status-box {
    background: $surface;
    margin: 1;
    padding: 1;
    border: tall $primary;
    height: auto;
}

/* Main Content */
#main {
    width: 1fr;
    height: 100%;
    padding: 1;
}

/* Forms */
.form-row {
    height: auto;
    margin-bottom: 1;
    align: center middle;
}

.form-label {
    width: 15;
    text-align: right;
    padding-right: 1;
}

Input {
    width: 20;
}

Select {
    width: 20;
}

/* Log Panel */
#log-panel {
    dock: bottom;
    height: 30%;
    border-top: solid $primary;
    background: $surface;
}

RichLog {
    background: $surface;
    color: $text;
}
"""

# ==============================================================================
# CUSTOM WIDGETS
# ==============================================================================

class LogSink:
    """Redirects Loguru messages to a Textual RichLog widget."""
    def __init__(self, widget: RichLog):
        self.widget = widget

    def write(self, message):
        self.widget.write(message)

# ==============================================================================
# MAIN APPLICATION
# ==============================================================================

class FactorLabApp(App):
    CSS = APP_CSS
    TITLE = "Factor Lab: Mission Control"
    BINDINGS = [
        ("q", "quit", "Quit"),
        ("d", "toggle_dark", "Toggle Dark Mode"),
        ("l", "toggle_log", "Toggle Logs"),
    ]

    def __init__(self):
        super().__init__()
        self.adapter = FactorLabAdapter()

    def compose(self) -> ComposeResult:
        # 1. Header
        yield Header()

        # 2. Body (Sidebar + Main)
        with Horizontal():
            # --- SIDEBAR ---
            with Vertical(id="sidebar"):
                yield Label("MODEL STATUS")
                with Vertical(id="status-box"):
                    yield Static("No Model Loaded", id="lbl-model-status")
                    yield Static("P: -  |  K: -", id="lbl-model-dims")
                
                yield Label("CONTROLS")
                yield Button("Reset Session", id="btn-reset", variant="error")

            # --- MAIN CONTENT ---
            with Vertical(id="main"):
                with TabbedContent(initial="tab-config"):
                    
                    # TAB 1: Configuration
                    with TabPane("1. Model Config", id="tab-config"):
                        yield Label("Define Factor Structure", classes="h1")
                        
                        with Horizontal(classes="form-row"):
                            yield Label("Assets (P):", classes="form-label")
                            yield Input("100", id="inp-p")
                        
                        with Horizontal(classes="form-row"):
                            yield Label("Factors (K):", classes="form-label")
                            yield Input("3", id="inp-k")
                        
                        yield Label("Generators", classes="h2")
                        # Simplified for TUI: Dropdowns for types
                        with Horizontal(classes="form-row"):
                            yield Label("Beta Dist:", classes="form-label")
                            yield Select([(x, x) for x in ["normal", "uniform", "student_t"]], id="sel-beta", value="normal")
                        
                        with Horizontal(classes="form-row"):
                            yield Label("Factor Vol:", classes="form-label")
                            yield Select([(x, x) for x in ["constant", "uniform"]], id="sel-fvol", value="constant")

                        with Horizontal(classes="form-row"):
                            yield Button("Build Generative Model", id="btn-build-gen", variant="primary")
                            yield Button("Run SVD Extraction (Demo)", id="btn-build-svd", variant="warning")

                    # TAB 2: Simulation
                    with TabPane("2. Simulation", id="tab-sim"):
                        yield Label("Monte Carlo Engine", classes="h1")
                        
                        with Horizontal(classes="form-row"):
                            yield Label("Horizon (N):", classes="form-label")
                            yield Input("252", id="inp-n")
                        
                        with Horizontal(classes="form-row"):
                            yield Label("Debug Rows:", classes="form-label")
                            yield Input("5", id="inp-debug")
                            yield Label("(Set > 0 to see Raw Data)", classes="dim")

                        yield Button("Run Simulation", id="btn-sim", variant="success")
                        
                        yield Label("Returns Preview (First 8 Assets)", classes="h2")
                        yield DataTable(id="table-returns")

                    # TAB 3: Optimization
                    with TabPane("3. Optimization", id="tab-opt"):
                        yield Label("Convex Optimization (SOCP)", classes="h1")
                        
                        with Horizontal(classes="form-row"):
                            yield Label("Long Only:", classes="form-label")
                            yield Switch(value=True, id="sw-long")
                        
                        yield Button("Solve Portfolio", id="btn-opt", variant="primary")
                        
                        yield Label("Results", classes="h2")
                        yield Static("No optimization run.", id="lbl-opt-result")

        # 3. Log Panel (Bottom Dock)
        with Vertical(id="log-panel"):
            yield Label("System Logs (Loguru)")
            yield RichLog(id="log-widget", markup=True)

        # 4. Footer
        yield Footer()

    def on_mount(self):
        """Setup Logging on startup."""
        log_widget = self.query_one("#log-widget", RichLog)
        
        # Configure Loguru to write to our widget
        logger.remove()
        logger.add(LogSink(log_widget), format="[{time:HH:mm:ss}] <level>{message}</level>", level="DEBUG")
        
        logger.info("Welcome to Factor Lab TUI.")
        logger.info("System initialized.")

    # --- ACTIONS ---

    @on(Button.Pressed, "#btn-build-gen")
    def action_build_gen(self):
        p = int(self.query_one("#inp-p", Input).value)
        k = int(self.query_one("#inp-k", Input).value)
        beta_dist = self.query_one("#sel-beta", Select).value
        fvol_dist = self.query_one("#sel-fvol", Select).value
        
        # In a real app, we'd have inputs for these params. Using defaults for TUI demo.
        params = {'normal': {'mean':0, 'std':1}, 'uniform': {'low':0, 'high':1}, 'student_t': {'df':4}, 'constant': {'c':0.1}}
        
        success = self.adapter.create_generative_model(
            p, k, 
            beta_dist, params.get(beta_dist, {}),
            fvol_dist, params.get(fvol_dist, {}),
            'constant', {'c': 0.05}
        )
        if success: self.update_status()

    @on(Button.Pressed, "#btn-build-svd")
    def action_build_svd(self):
        p = int(self.query_one("#inp-p", Input).value)
        k = int(self.query_one("#inp-k", Input).value)
        success = self.adapter.create_svd_model(1000, p, k) # Simulate 1000 days history
        if success: self.update_status()

    # FIX: Added 'thread=True' to allow synchronous numpy/simulation code
    @on(Button.Pressed, "#btn-sim")
    @work(exclusive=True, thread=True) 
    def action_simulate(self):
        n = int(self.query_one("#inp-n", Input).value)
        debug_len = int(self.query_one("#inp-debug", Input).value)
        
        success = self.adapter.run_simulation(n, debug_len)
        
        if success:
            # Update Table (Call back to main thread for UI updates)
            self.call_from_thread(self.update_returns_table)

    def update_returns_table(self):
        table = self.query_one("#table-returns", DataTable)
        table.clear(columns=True)
        
        data = self.adapter.get_returns_preview()
        if not data: return
        
        # Add Columns (Time + Asset 0..N)
        cols = ["Time"] + [f"A{i}" for i in range(len(data[0])-1)]
        table.add_columns(*cols)
        table.add_rows(data)

    @on(Button.Pressed, "#btn-opt")
    def action_optimize(self):
        long_only = self.query_one("#sw-long", Switch).value
        success = self.adapter.optimize_portfolio(long_only=long_only)
        
        res_lbl = self.query_one("#lbl-opt-result", Static)
        if success:
            res_lbl.update(self.adapter.get_optimization_summary())
        else:
            res_lbl.update("[bold red]Optimization Failed.[/]")

    def update_status(self):
        """Reflect Adapter state in Sidebar."""
        if self.adapter.model:
            self.query_one("#lbl-model-status").update("[bold green]Active[/]")
            self.query_one("#lbl-model-dims").update(f"P: {self.adapter.p_assets} | K: {self.adapter.k_factors}")
        else:
            self.query_one("#lbl-model-status").update("[bold red]No Model[/]")

if __name__ == "__main__":
    app = FactorLabApp()
    app.run()