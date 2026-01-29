# Make a Factor Model with these Parameters

p = 500 = number of securities

## betas = B with shape (p,2)

### Column 1 = factor 1 :

beta[0] = 500 draws from N(1, 0.25)

### Column 2 = factor 2:

beta[1].temporary = 500 draws preliminary exposures from N(0,1)
 beta[1].final = beta[1].temporary orthogonalized against factor 1 via Gram-Schmidt orthogonalization

beta[1] = beta[1].final

## Factor variances:

diagonal with entries (0.18^2, 0.05^2)

## Specific Covariance matrix with shape (p,p)

Diagonal matrix with all diagonal entries = (0.16)

# Simulate 63 returns from this Factor model

1. using Gaussians with mean 0 and the variances above.

2. Using mean 0 t-distributions with 5 degrees of freedom for factor returns and 4 degrees of freedom for specific returns, and the variances above.

p = 500 = number of securities

## betas = B with shape (p,2)

### Column 1 = factor 1 :

beta[0] = 500 draws from N(1, 0.25)

### Column 2 = factor 2:

beta[1].temporary  = 500 draws preliminary exposures from N(0,1)
 beta[1].final = beta[1].temporary orthogonalized against factor 1 via Gram-Schmidt orthogonalization

beta[1] = beta[1].final

## Factor variances:

diagonal with entries (0.18^2, 0.05^2)

## Specific Covariance matrix with shape (p,p)

Diagonal matrix with all diagonal entries = (0.16)  

# Simulate 63 returns from this Factor model

1. using Gaussians with mean 0 and the variances above.

2. Using mean 0 t-distributions with 5 degrees of freedom for factor returns and 4 degrees of freedom for specific returns, and the variances above.




