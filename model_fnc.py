import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from collections import OrderedDict
import utility_fnc
from PIL import Image


def get_dataloaders(data_dir, use_gpu, pin_memory):
    ''' Return dataloaders for training, validation and testing datasets.
    Return a dictionary to map indexes to classes.
    '''
    # set data paths
    train_dir = data_dir + "/train"
    valid_dir = data_dir + "/valid"
    test_dir = data_dir + "/test"

    # kwargs = {'num_workers': num_workers, 'pin_memory': pin_memory}
    kwargs = {'pin_memory': pin_memory}

    network_means = [0.485, 0.456, 0.406]
    network_std = [0.229, 0.224, 0.225]
    IMAGE_SIZE = 224
    BATCH_SIZE = 64

    # Define transforms for the training, validation, and testing sets
    data_transforms = {
        "training": transforms.Compose([transforms.RandomVerticalFlip(),
                                        transforms.RandomRotation(30),
                                        transforms.RandomResizedCrop(IMAGE_SIZE),
                                        transforms.ToTensor(),
                                        transforms.Normalize(network_means, network_std)]),

        "validation": transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(IMAGE_SIZE),
                                          transforms.ToTensor(),
                                          transforms.Normalize(network_means, network_std)]),

        "testing": transforms.Compose([transforms.Resize(256),
                                       transforms.CenterCrop(IMAGE_SIZE),
                                       transforms.ToTensor(),
                                       transforms.Normalize(network_means, network_std)])

    }

    # Load the datasets with ImageFolder
    image_datasets = {
        "training": datasets.ImageFolder(train_dir, transform=data_transforms["training"]),
        "validation": datasets.ImageFolder(valid_dir, transform=data_transforms["validation"]),
        "testing": datasets.ImageFolder(test_dir, transform=data_transforms["testing"])
    }

    # Using the image datasets and the transforms, define the dataloaders
    dataloaders = {
        "training": torch.utils.data.DataLoader(image_datasets["training"], batch_size=BATCH_SIZE, shuffle=True, **kwargs),
        "validation": torch.utils.data.DataLoader(image_datasets["validation"], batch_size=BATCH_SIZE, shuffle=True, **kwargs),
        "testing": torch.utils.data.DataLoader(image_datasets["testing"], batch_size=BATCH_SIZE, shuffle=False, **kwargs)
    }

    class_to_idx = image_datasets['training'].class_to_idx

    return dataloaders, class_to_idx


def get_model_from_arch(arch, hidden_units):
    ''' Load an existing PyTorch model, freeze parameters and substitute classifier.
    '''

    if arch == "vgg13":
        model = models.vgg13(pretrained=True)
        input_size = model.classifier[0].in_features
    else:
        raise RuntimeError("unknown model")

    # prevent backpropagation on network parameters by freezing parameters
    for param in model.parameters():
        param.requires_grad = False

    # setting "output_size" to 102
    output_size = 102

    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_size, hidden_units)),
        ('relu', nn.ReLU()),
        ('dropout', nn.Dropout(p=0.1)),
        ('fc2', nn.Linear(hidden_units, output_size)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    model.classifier = classifier

    return model



def create_new_model(arch, learning_rate, hidden_units, class_to_idx):

    # load the pretrained vgg model
    model = get_model_from_arch(arch, hidden_units)
    # set training parameters
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    # using the Adam optimizer
    optimizer = optim.Adam(parameters, lr=learning_rate)
    optimizer.zero_grad()
    # loss criterion is "negative log likelihood loss" which is useful to train classification problems
    criterion = nn.NLLLoss()
    # save class to index mapping
    model.class_to_idx = class_to_idx

    return model, optimizer, criterion


def save_checkpoint(checkpoint_path, model, optimizer, arch, learning_rate, hidden_units, EPOCHS):
    ''' Save the checkpoint
    '''
    state = {
        'arch': arch,
        'learning_rate': learning_rate,
        'hidden_units': hidden_units,
        'epochs': EPOCHS,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'class_to_idx': model.class_to_idx
    }

    torch.save(state, checkpoint_path)

    print("Checkpoint Saved: '{}'".format(checkpoint_path))


def load_checkpoint(checkpoint_path):
    ''' Load a previously trained deep learning model checkpoint
    '''
    state = torch.load(checkpoint_path)

    # create pretrained model
    model, optimizer, criterion = create_new_model(state['arch'],
                                               state['learning_rate'],
                                               state['hidden_units'],
                                               state['class_to_idx'])

    # load checkpoint state into the model
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])

    print("Loaded: '{}' (arch={}, hidden_units={}, epochs={})".format(
            checkpoint_path, state['arch'], state['hidden_units'], state['epochs']))

    return model



def validate_model(model, criterion, data_loader, use_gpu):
    # model in inference mode
    model.eval()
    # initiate the accuracy and loss at zero
    accuracy = 0
    loss = 0

    for images, labels in iter(data_loader):

        with torch.no_grad():
            if use_gpu:
                images = images.float().cuda()
                labels = labels.long().cuda()
            else:
                images = images
                labels = labels

        output = model.forward(images)
        loss += criterion(output, labels).item()

        # since the model's output is log-softmax, taking exponentials to get the probabilities of classes
        prob = torch.exp(output).data

        # class with highest probability is the predicted class
        prediction = (labels.data == prob.max(1)[1])

        # accuracy is the number of correct predictions divided by all predictions
        accuracy += prediction.type_as(torch.FloatTensor()).mean()

    valid_loss = loss/len(data_loader)
    valid_acc = accuracy/len(data_loader)

    return valid_loss, valid_acc


def train_model(model, criterion, optimizer, epochs, training_data_loader, validation_data_loader, use_gpu):
    # model in training mode
    model.train()

    # set the batch sizes inside epochs
    print_every = 40
    steps = 0

    for epoch in range(epochs):
        running_loss = 0

        for images, labels in iter(training_data_loader):
            steps += 1

            # move tensors to GPU if available
            with torch.no_grad():
                if use_gpu:
                    images = images.float().cuda()
                    labels = labels.long().cuda()
                else:
                    images = images
                    labels = labels

            # set gradients to zero
            optimizer.zero_grad()
            # do a forward pass
            output = model.forward(images)
            # calculate loss
            loss = criterion(output, labels)
            # do a backward pass
            loss.backward()
            # update weights using the Adam optimizer
            optimizer.step()
            # update the running total of loss
            running_loss += loss.item()

            if steps % print_every == 0:
                validation_loss, validation_accuracy = validate_model(model, criterion, validation_data_loader, use_gpu)

                print("EPOCH: {}/{}\t".format(epoch+1, epochs),
                      "Training Loss: {:.3f}\t".format(running_loss/print_every),
                      "Validation Loss: {:.3f}\t".format(validation_loss),
                      "Validation Accuracy: {:.3f}".format(validation_accuracy))

                # set the running total back to zero for the next epoch
                running_loss = 0

                # set the model back into training mode
                model.train()



def predict(image_path, model, use_gpu, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    # TODO: Implement the code to predict the class from an image file

    # set model in inference mode
    model.eval()

    image = Image.open(image_path)
    np_array = utility_fnc.process_image(image)
    tensor = torch.from_numpy(np_array)

    # use GPU if available
    if use_gpu:
        tensor_V = tensor.float().cuda()
    else:
        tensor_V = tensor.float()

    # add a new dimension as the model is expecting a 4d tensor
    tensor_V = tensor_V.unsqueeze(0)

    # "with torch.no_grad():" apparently serves the same purpose as "Variable(tensor, volatile=True)"
    # disabling gradient calculation
    with torch.no_grad():
        # do a forward run through the model
        output = model.forward(tensor_V)

        # since the model's output is log-softmax, taking exponentials to get the probabilities of classes
        #probs, classes = torch.exp(output).data.topk(topk)
        probs, classes = torch.topk(output, topk)
        probs = probs.exp()

        # Move results to CPU if needed
        probs = probs.cpu() if use_gpu else probs
        classes = classes.cpu() if use_gpu else classes

    # map classes to indices
    inverted_class_to_idx = {
        model.class_to_idx[k]: k for k in model.class_to_idx}

    mapped_classes = list()

    for label in classes.numpy()[0]:
        mapped_classes.append(inverted_class_to_idx[label])

    return probs.numpy()[0], mapped_classes
