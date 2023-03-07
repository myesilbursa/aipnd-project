import argparse
from time import time
import torch
import utility_fnc
import model_fnc
import os


def get_input_args():
    
    archs = {"vgg13"}
    
    parser = argparse.ArgumentParser()

    # Add positional arguments
    parser.add_argument('data_dir', type=str,
                        help='Directory used to locate source images')

    # Add optional arguments
    parser.add_argument('--save_dir', type=str,
                        help='Directory used to save checkpoints')

    parser.add_argument('--arch', dest='arch', default='vgg', action='store', choices=archs,
                        help='Model architecture to use for training')

    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate hyperparameter')

    parser.add_argument('--hidden_units', type=int, default=400,
                        help='Number of hidden units hyperparameter')

    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of epochs used to train model')

    parser.add_argument('--gpu', dest='gpu', action='store_true', 
                        help='Use GPU for training')
    parser.set_defaults(gpu=False)

    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of subprocesses to use for data loading')

    parser.add_argument('--pin_memory', dest='pin_memory', action='store_true', 
                        help='Request data loader to copy tensors into CUDA pinned memory')
    parser.set_defaults(pin_memory=False)

    parser.add_argument('--num_threads', type=int, default=16,
                        help='Number of threads used to train model when using CPU')

    return parser.parse_args()


def main():
    start_time = time()

    in_args = get_input_args()

    # check for GPU
    use_gpu = torch.cuda.is_available() and in_args.gpu

    # print parameter information
    if use_gpu:
        print("Training on GPU{}".format(" with pinned memory" if in_args.pin_memory else "."))
    else:
        print("Training on CPU using {} threads.".format(in_args.num_threads))

    print("Architecture:{}, Learning rate:{}, Hidden Units:{}, Epochs:{}".format(
        in_args.arch, in_args.learning_rate, in_args.hidden_units, in_args.epochs))

    # Get dataloaders for training
    dataloaders, class_to_idx = model_fnc.get_dataloaders(in_args.data_dir,
                                                            use_gpu,
                                                            in_args.pin_memory)

    # Create model
    model, optimizer, criterion = model_fnc.create_new_model(in_args.arch,
                                                            in_args.learning_rate,
                                                            in_args.hidden_units,
                                                            class_to_idx)

    # move tensors to GPU if available
    if use_gpu:
        model.cuda()
        criterion.cuda()
    else:
        torch.set_num_threads(in_args.num_threads)

    # train the network
    model_fnc.train_model(model, criterion, optimizer, in_args.epochs,
                    dataloaders['training'], dataloaders['validation'], use_gpu)

    # save trained model
    if in_args.save_dir:
        # create save directory if required
        if not os.path.exists(in_args.save_dir):
            os.makedirs(in_args.save_dir)

        # save checkpoint in save directory
        file_path = in_args.save_dir + '/' + in_args.arch + '_checkpoint.pth'
    else:
        # save checkpoint in current directory
        file_path = in_args.arch + '_checkpoint.pth'

    model_fnc.save_checkpoint(file_path, model, optimizer,
                            in_args.arch, in_args.learning_rate,
                            in_args.hidden_units, in_args.epochs)

    # get prediction accuracy using test dataset
    testing_loss, testing_accuracy = model_fnc.validate_model(model, criterion, dataloaders['testing'], use_gpu)
    print("Testing Accuracy: {:.3f}".format(testing_accuracy))

    # computes overall runtime in seconds & prints it in hh:mm:ss format
    end_time = time()
    utility_fnc.print_elapsed_time(end_time - start_time)


# Call to main function to run the program
if __name__ == "__main__":
    main()
