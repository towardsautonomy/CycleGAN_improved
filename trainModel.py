import torch.optim as optim
from torchsummary import summary
import matplotlib.pyplot as plt
from collections import deque
import math
import time
import csv

from dataLoader import GAN_DataLoader, imshow
from model import CycleGAN
from lossFunc import *
from utils import *
from config import *

# configure full paths
checkpoints_dir = os.path.join(checkpoints_dir, experiment_id)
samples_dir = os.path.join(samples_dir, experiment_id)
logs_dir = os.path.join(logs_dir, experiment_id)

# make directories
os.system('mkdir -p '+checkpoints_dir)
os.system('mkdir -p '+samples_dir)
os.system('mkdir -p '+logs_dir)

## create models
# call the function to get models
G_XtoY, G_YtoX, D_X, D_Y = CycleGAN()

# define optimizer parameters
g_params = list(G_XtoY.parameters()) + list(G_YtoX.parameters())  # Get generator parameters

# Create optimizers for the generators and discriminators
g_optimizer = optim.Adam(g_params, lr, [beta1, beta2])
d_x_optimizer = optim.Adam(D_X.parameters(), lr, [beta1, beta2])
d_y_optimizer = optim.Adam(D_Y.parameters(), lr, [beta1, beta2])

# count number of parameters in a model
def count_model_parameters(model):
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_non_trainable_params = n_params - n_trainable_params

    return n_params, n_trainable_params, n_non_trainable_params
    
# training code
def trainModel(dloader_train_it, dloader_test_it, n_train_batch_per_epoch, n_test_batch_per_epoch, batch_size = 8, n_epochs=1000):
    n_consecutive_epochs=0 # number of consecutive epochs for early stop
    mean_epochs=10 # number of epochs to average the scores
    
    # keep track of losses over time
    losses = []
    min_loss = 1.0e6

    # Get some fixed data from domains X and Y for sampling. These are images that are held
    # constant throughout training, that allow us to inspect the model's performance.
    fixed_X, fixed_Y = next(dloader_test_it)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # open log file in append mode
    log_file = logs_dir+'/log.csv'
    # check if the log file exists
    logfile_exists = os.path.exists(log_file)
    with open(log_file, 'a+', newline='') as csvfile:
        fieldnames = ['epoch',                \
                      'd_X_loss',             \
                      'd_Y_loss',             \
                      'recon_X_loss',         \
                      'recon_Y_loss',         \
                      'total_loss',           \
                      'valid_d_X_loss',       \
                      'valid_d_Y_loss',       \
                      'valid_recon_X_loss',   \
                      'valid_recon_Y_loss',   \
                      'valid_total_loss'      ]
                      
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        # write header if the file did not exist earlier
        if not logfile_exists:
            writer.writeheader()

        # iterate over number of epochs
        for epoch in range(init_epoch, n_epochs):
            train_losses = deque(maxlen=mean_epochs)
            start_time = time.time()
            # iterate over all the batches in a single epoch
            for iteration in range(0, n_train_batch_per_epoch):
                images_X, images_Y = next(dloader_train_it)
                
                # move images to GPU if available
                images_X = images_X.to(device)
                images_Y = images_Y.to(device)

                '''
                Train the Discriminators 
                 - Get discriminator loss on real images
                 - Get discriminator loss on fake images
                 - Add the losses
                 - Perform back-propagation
                '''
                ## Discriminators on Domain X
                # Train with real images
                d_x_optimizer.zero_grad()

                # Compute the discriminator losses on real images
                out_x = D_X(images_X)
                D_X_real_loss = real_discriminator_loss(out_x, lambda_weight=lambda_discriminator)
                
                # Train with fake images
                
                # Generate fake images that look like domain X based on real images in domain Y
                fake_X = G_YtoX(images_Y)

                # Compute the fake loss for D_X
                out_x = D_X(fake_X)
                D_X_fake_loss = fake_discriminator_loss(out_x, lambda_weight=lambda_discriminator)
                
                # Compute the total loss and perform backprop
                d_x_loss = D_X_real_loss + D_X_fake_loss
                d_x_loss.backward()
                d_x_optimizer.step()

                ## Discriminators on Domain Y
                # Train with real images
                d_y_optimizer.zero_grad()
                
                # Compute the discriminator losses on real images
                out_y = D_Y(images_Y)
                D_Y_real_loss = real_discriminator_loss(out_y, lambda_weight=lambda_discriminator)
                
                # Train with fake images

                # Generate fake images that look like domain Y based on real images in domain X
                fake_Y = G_XtoY(images_X)

                # Compute the fake loss for D_Y
                out_y = D_Y(fake_Y)
                D_Y_fake_loss = fake_discriminator_loss(out_y, lambda_weight=lambda_discriminator)

                # Compute the total loss and perform backprop
                d_y_loss = D_Y_real_loss + D_Y_fake_loss
                d_y_loss.backward()
                d_y_optimizer.step()

                '''
                Train the Generator
                 - Generate X from Y
                 - Apply Discriminator
                 - Compute Discriminator Loss
                 - Reconstruct X from the generated Y
                 - Compute Cycle-Consistency Loss

                 - Generate Y from X
                 - Apply Discriminator
                 - Compute Discriminator Loss
                 - Reconstruct Y from the generated X
                 - Compute Cycle-Consistency Loss

                 - Add the losses
                 - Perform back-propagation
                '''

                ## generate fake X images and reconstructed Y images
                g_optimizer.zero_grad()

                # Generate fake images that look like domain X based on real images in domain Y
                fake_X = G_YtoX(images_Y)

                # Compute the generator loss based on domain X
                out_x = D_X(fake_X)
                g_YtoX_loss = real_discriminator_loss(out_x, lambda_weight=lambda_discriminator)

                # Create a reconstructed y
                # Compute the cycle consistency loss (the reconstruction loss)
                reconstructed_Y = G_XtoY(fake_X)
                reconstructed_y_loss = cycle_consistency_loss(images_Y, reconstructed_Y, lambda_weight=lambda_cycle_consistency)


                ## generate fake Y images and reconstructed X images

                # Generate fake images that look like domain Y based on real images in domain X
                fake_Y = G_XtoY(images_X)

                # Compute the generator loss based on domain Y
                out_y = D_Y(fake_Y)
                g_XtoY_loss = real_discriminator_loss(out_y, lambda_weight=lambda_discriminator)

                # Create a reconstructed x
                # Compute the cycle consistency loss (the reconstruction loss)
                reconstructed_X = G_YtoX(fake_Y)
                reconstructed_x_loss = cycle_consistency_loss(images_X, reconstructed_X, lambda_weight=lambda_cycle_consistency)

                # Add up all generator and reconstructed losses and perform backprop
                g_total_loss = g_XtoY_loss + g_YtoX_loss + reconstructed_x_loss + reconstructed_y_loss
                g_total_loss.backward()
                g_optimizer.step()

                # Print the log info
                # append real and fake discriminator losses and the generator loss
                train_losses.append([d_x_loss.item(), d_y_loss.item(), reconstructed_x_loss.item(), reconstructed_y_loss.item(), g_total_loss.item()])
                print('\rEpoch [{:5d}/{:5d}] | Iteration [{:5d}/{:5d}] | d_X_loss: {:6.4f} | d_Y_loss: {:6.4f} | recon_X_loss: {:6.4f} | recon_Y_loss: {:6.4f} | total_loss: {:6.4f}'.format(
                        epoch, n_epochs, iteration, n_train_batch_per_epoch, d_x_loss.item(), d_y_loss.item(), reconstructed_x_loss.item(), reconstructed_y_loss.item(), g_total_loss.item()), end="")

            # compute time taken for training this batch
            time_taken = (time.time() - start_time)

            # perform test on validation samples
            validation_losses = deque(maxlen=mean_epochs)
            for iteration in range(0, n_test_batch_per_epoch):
                images_X, images_Y = next(dloader_test_it)

                # move images to GPU if available
                images_X = images_X.to(device)
                images_Y = images_Y.to(device)

                # set models to eval mode for validation
                G_YtoX.eval() 
                G_XtoY.eval()
                D_X.eval()
                D_Y.eval()

                # X->Y->Reconstructed X
                fake_Y = G_XtoY(images_X)
                recon_Y_X = G_YtoX(fake_Y)
                reconstructed_X_loss = cycle_consistency_loss(images_X, recon_Y_X, lambda_weight=lambda_cycle_consistency)

                # Y->X->Reconstructed Y
                fake_X = G_YtoX(images_Y)
                recon_X_Y = G_XtoY(fake_X)
                reconstructed_Y_loss = cycle_consistency_loss(images_Y, recon_X_Y, lambda_weight=lambda_cycle_consistency)

                # Discriminator X loss
                disc_X_loss = real_discriminator_loss(D_X(images_X), lambda_weight=lambda_discriminator) + \
                                real_discriminator_loss(D_X(fake_X), lambda_weight=lambda_discriminator)

                # Discriminator Y loss
                disc_Y_loss = real_discriminator_loss(D_Y(images_Y), lambda_weight=lambda_discriminator) + \
                                real_discriminator_loss(D_Y(fake_Y), lambda_weight=lambda_discriminator)

                # total validation loss
                total_valid_loss = reconstructed_X_loss + reconstructed_Y_loss + disc_X_loss + disc_Y_loss

                # append to list
                validation_losses.append([disc_X_loss.item(), disc_Y_loss.item(), reconstructed_X_loss.item(), reconstructed_Y_loss.item(), total_valid_loss.item()])

                # set models back to train mode
                G_YtoX.train()
                G_XtoY.train()
                D_X.eval()
                D_Y.eval()

            ## Compute average of losses
            # training losses
            train_losses = np.array(train_losses)
            train_d_X_loss = np.mean(train_losses[:,0])
            train_d_Y_loss = np.mean(train_losses[:,1])
            train_recon_X_loss = np.mean(train_losses[:,2])
            train_recon_Y_loss = np.mean(train_losses[:,3])
            train_total_G_loss = np.mean(train_losses[:,4])
            # validation losses
            validation_losses = np.array(validation_losses)
            valid_d_X_loss = np.mean(validation_losses[:,0])
            valid_d_Y_loss = np.mean(validation_losses[:,1])
            valid_recon_X_loss = np.mean(validation_losses[:,2])
            valid_recon_Y_loss = np.mean(validation_losses[:,3])
            valid_total_G_loss = np.mean(validation_losses[:,4])
            print('\rEpoch [{:5d}/{:5d}] Training Losses   | d_X_loss: {:6.4f} | d_Y_loss: {:6.4f} | recon_X_loss: {:6.4f} | recon_Y_loss: {:6.4f} | total_loss: {:6.4f} | Time Taken: {:.2f} sec'.format(
                        epoch, n_epochs, train_d_X_loss, train_d_Y_loss, train_recon_X_loss, train_recon_Y_loss, train_total_G_loss, time_taken))
            print('\rEpoch [{:5d}/{:5d}] Validation Losses | d_X_loss: {:6.4f} | d_Y_loss: {:6.4f} | recon_X_loss: {:6.4f} | recon_Y_loss: {:6.4f} | total_loss: {:6.4f}'.format(
                        epoch, n_epochs, valid_d_X_loss, valid_d_Y_loss, valid_recon_X_loss, valid_recon_Y_loss, valid_total_G_loss))

            # Save the generated samples
            if epoch % sample_every == 0:
                # set generators to eval mode for sample generation
                G_YtoX.eval() 
                G_XtoY.eval()
                save_samples(samples_dir, epoch, fixed_Y, fixed_X, G_YtoX, G_XtoY, batch_size=batch_size)
                # set generators back to train mode
                G_YtoX.train()
                G_XtoY.train()

            # Save the model parameters
            if epoch % checkpoint_every == 0:
                checkpoint(checkpoints_dir, epoch, G_XtoY, G_YtoX, D_X, D_Y)

            # save best weights
            if train_total_G_loss <= min_loss:
                print('Total Loss decreased from {:.5f} to {:.5f}. Saving Model.'.format(min_loss, train_total_G_loss))
                checkpoint(checkpoints_dir, epoch, G_XtoY, G_YtoX, D_X, D_Y, best=True)
                min_loss = train_total_G_loss
                n_consecutive_epochs = 0
            else:
                print('Total Loss did not improve from {:.5f}'.format(min_loss))
                n_consecutive_epochs += 1

            # log to the csv file
            writer.writerow({'epoch': epoch,                            \
                             'd_X_loss'          : train_d_X_loss,      \
                             'd_Y_loss'          : train_d_Y_loss,      \
                             'recon_X_loss'      : train_recon_X_loss,  \
                             'recon_Y_loss'      : train_recon_Y_loss,  \
                             'total_loss'        : train_total_G_loss,  \
                             'valid_d_X_loss'    : valid_d_X_loss,      \
                             'valid_d_Y_loss'    : valid_d_Y_loss,      \
                             'valid_recon_X_loss': valid_recon_X_loss,  \
                             'valid_recon_Y_loss': valid_recon_Y_loss,  \
                             'valid_total_loss'  : valid_total_G_loss})

            # early stop
            if n_consecutive_epochs >= early_stop_epoch_thres:
                print('Total Loss did not improve for {} consecutive epochs. Early Stopping!'.format(n_consecutive_epochs))
                break
        
    return losses

if __name__ == '__main__':
    # print model summary
    print('==========================================================')
    print('=====================  Model Summary  ====================')
    print('==========================================================')
    n_params, n_trainable_params, n_non_trainable_params = count_model_parameters(G_XtoY)
    print('G_XtoY:')
    print('\t- Num of Parameters                : {:,}'.format(n_params))
    print('\t- Num of Trainable Parameters      : {:,}'.format(n_trainable_params))
    print('\t- Num of Non-Trainable Parameters  : {:,}'.format(n_non_trainable_params))
    n_params, n_trainable_params, n_non_trainable_params = count_model_parameters(G_YtoX)
    print('G_YtoX:')
    print('\t- Num of Parameters                : {:,}'.format(n_params))
    print('\t- Num of Trainable Parameters      : {:,}'.format(n_trainable_params))
    print('\t- Num of Non-Trainable Parameters  : {:,}'.format(n_non_trainable_params))
    n_params, n_trainable_params, n_non_trainable_params = count_model_parameters(D_X)
    print('D_X:')
    print('\t- Num of Parameters                : {:,}'.format(n_params))
    print('\t- Num of Trainable Parameters      : {:,}'.format(n_trainable_params))
    print('\t- Num of Non-Trainable Parameters  : {:,}'.format(n_non_trainable_params))
    n_params, n_trainable_params, n_non_trainable_params = count_model_parameters(D_Y)
    print('D_Y:')
    print('\t- Num of Parameters                : {:,}'.format(n_params))
    print('\t- Num of Trainable Parameters      : {:,}'.format(n_trainable_params))
    print('\t- Num of Non-Trainable Parameters  : {:,}'.format(n_non_trainable_params))
    print('==========================================================')
    summary(G_XtoY, (3,image_size[0],image_size[1]))

    # load weights
    if use_pretrained_weights:
        G_XtoY.load_state_dict(torch.load(generator_x_y_weights, map_location=lambda storage, loc: storage))
        G_YtoX.load_state_dict(torch.load(generator_y_x_weights, map_location=lambda storage, loc: storage))
        D_X.load_state_dict(torch.load(discriminator_x_weights, map_location=lambda storage, loc: storage))
        D_Y.load_state_dict(torch.load(discriminator_y_weights, map_location=lambda storage, loc: storage))
        print('Loaded pretrained weights')
    
    # get data loaders
    dloader = GAN_DataLoader(imageX_dir=domain_a_dir, imageY_dir=domain_b_dir, image_size=image_size)

    dloader_train, dloader_test = dloader.get_data_generator(n_samples=n_samples, test_size=test_size, batch_size=batch_size)
    dloader_train_it = iter(dloader_train)
    dloader_test_it = iter(dloader_test)

    # compute number of iterations per epoch
    n_train_samples, n_test_samples = dloader.get_num_samples(n_samples=n_samples, test_size=test_size)
    n_train_batch_per_epoch = math.ceil(n_train_samples/batch_size)
    n_test_batch_per_epoch = math.ceil(n_test_samples/batch_size)
    print('Training on {} samples and testing on {} samples for a maximum of {} epochs.'.format(n_train_samples, n_test_samples, n_epochs))

    losses = trainModel(dloader_train_it, dloader_test_it, n_train_batch_per_epoch, n_test_batch_per_epoch, batch_size=batch_size, n_epochs=n_epochs)

    fig, ax = plt.subplots(figsize=(12,8))
    losses = np.array(losses)
    plt.plot(losses.T[0], label='Discriminator, X', alpha=0.5)
    plt.plot(losses.T[1], label='Discriminator, Y', alpha=0.5)
    plt.plot(losses.T[2], label='Generators', alpha=0.5)
    plt.title("Training Losses")
    plt.legend()