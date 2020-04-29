import torch

## Define Loss Functions
# discriminator patch loss for real samples
def real_discriminator_loss(D_out, lambda_weight=1.0):
    # how close is the produced output from being "real"?
    return lambda_weight*torch.mean((D_out-1)**2)

# discriminator patch loss for fake samples
def fake_discriminator_loss(D_out, lambda_weight=1.0):
    # how close is the produced output from being "fake"?
    return lambda_weight*torch.mean(D_out**2)

# cycle consistency loss
def cycle_consistency_loss(real_im, reconstructed_im, lambda_weight=1.0):
    # calculate reconstruction loss 
    # as absolute value difference between the real and reconstructed images
    reconstr_loss = torch.mean(torch.abs(real_im - reconstructed_im))
    # return weighted loss
    return lambda_weight*reconstr_loss