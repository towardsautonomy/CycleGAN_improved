## hyper parameters
lr=0.0002                       # Learning Rate
beta1=0.5                       # Beta 1 for Adam optimizer
beta2=0.999                     # Beta 2 for Adam optimizer
    
batch_size=8                    # batch size
init_epoch=0                    # initial epoch
n_epochs=200                    # maximum nomber of epochs
n_samples=-1                    # -1 for all available samples
test_size=0.05                  # fraction of all samples for validation
early_stop_epoch_thres=10       # threshold for stopping training if loss does not improve
    
image_size = (256, 256)         # image size 

# lambda for losses
lambda_discriminator=1.0        # lambda weight for discriminator loss
lambda_cycle_consistency=1.0    # lambda weight for cycle-consistency loss

# flags
use_pretrained_weights=False

# experiment id
experiment_id = 'big_model'

## paths
domain_a_dir = '/home/shubham/workspace/dataset/vKITTI/*/clone/frames/rgb/*/*.jpg'
domain_b_dir = '/home/shubham/workspace/dataset/KITTI/data_object_image_2/*/*/*.png'

# directories
checkpoints_dir = 'checkpoints'
samples_dir = 'samples'
logs_dir = 'logs'

# pretrained weights
generator_x_y_weights = './checkpoints/big_model/best/G_XtoY.pkl'
generator_y_x_weights = './checkpoints/big_model/best/G_YtoX.pkl'
discriminator_x_weights = './checkpoints/big_model/best/D_X.pkl'
discriminator_y_weights = './checkpoints/big_model/best/D_Y.pkl'

## Number of epochs to sample and checkpoint
sample_every=1
checkpoint_every=10