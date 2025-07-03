# This serves as a template which will guide you through the implementation of this task.
# It is advised to first read the whole template and get a sense of the overall
# structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
import os
import torch
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# The device is automatically set to GPU if available, otherwise CPU
# If you want to force the device to CPU, you can change the line to
# device = torch.device("cpu")
# When using the GPU, it is important that your model and all data are on the
# same device.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Global Variables
global_batch_size = 128
global_num_workers = 16
global_model = models.resnet18(weights='IMAGENET1K_V1')
global_embedding_size = 512

# Debug
verbose = False
testing = False


def generate_embeddings():
    """
    Transform, resize and normalize the images and then use a pretrained model to extract
    the embeddings.
    """

    # TODO: define a transform to pre-process the images
    # The required pre-processing depends on the pre-trained model you choose below.
    # See https://pytorch.org/vision/stable/models.html#using-the-pre-trained-models
    train_transforms = transforms.Compose([  # Transform from explanation + transfer learning
                                            # transforms.RandomResizedCrop(224),
                                            # transforms.RandomHorizontalFlip(),
                                            transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                            ])

    # Accepts PIL.Image, batched (B, C, H, W) and single (C, H, W) image torch.Tensor objects. The images are resized to resize_size=[256] using interpolation=InterpolationMode.BILINEAR, followed by a central crop of crop_size=[224]. Finally the values are first rescaled to [0.0, 1.0] and then normalized using mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225].

    train_dataset = datasets.ImageFolder(root="dataset/", transform=train_transforms)

    # Hint: adjust batch_size and num_workers to your PC configuration, so that you don't
    # run out of memory (VRAM if on GPU, RAM if on CPU)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=global_batch_size,
                              shuffle=False,
                              pin_memory=True,
                              num_workers=global_num_workers)

    # TODO: define a model for extraction of the embeddings (Hint: load a pretrained model,
    # more info here: https://pytorch.org/vision/stable/models.html)
    model = global_model
    model.to(device)
    embedding_size = global_embedding_size  # Dummy variable, replace with the actual embedding size once you pick your model
    num_images = len(train_dataset)
    embeddings = np.zeros((num_images, embedding_size))

    # TODO: Use the model to extract the embeddings. Hint: remove the last layers of the
    # model to access the embeddings the model generates.

    # Pinched of stackoverflow
    # strip the last layer
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])

    with torch.no_grad():
        start_idx = 0
        for inputs, _ in train_loader:
            inputs = inputs.to(device)
            outputs = feature_extractor(inputs)
            print(outputs.shape)
            embed = outputs.resize_(outputs.shape[0], outputs.shape[1])
            print(embed.shape)
            batch_size = outputs.shape[0]
            embeddings[start_idx:start_idx+batch_size] = outputs
            start_idx += batch_size
            print(start_idx)

        # output = feature_extractor(input)
        # embed = output.resize_(512)

    print(embeddings.shape)
    print(embeddings)

    np.save('dataset/embeddings.npy', embeddings)


def get_data(file, train=True):
    """
    Load the triplets from the file and generate the features and labels.

    input: file: string, the path to the file containing the triplets
          train: boolean, whether the data is for training or testing

    output: X: numpy array, the features
            y: numpy array, the labels
    """
    triplets = []
    with open(file) as f:
        for line in f:
            triplets.append(line)

    # generate training data from triplets
    train_dataset = datasets.ImageFolder(root="dataset/",
                                         transform=None)
    filenames = [s[0].split('/')[-1].replace('.jpg', '') for s in train_dataset.samples]
    embeddings = np.load('dataset/embeddings.npy')
    # TODO: Normalize the embeddings

    # off of Rasim's program
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

    file_to_embedding = {}
    for i in range(len(filenames)):
        file_to_embedding[filenames[i]] = embeddings[i]
    X = []
    y = []
    # use the individual embeddings to generate the features and labels for triplets
    for t in triplets:
        emb = [file_to_embedding[a] for a in t.split()]
        X.append(np.hstack([emb[0], emb[1], emb[2]]))
        y.append(1)
        # Generating negative samples (data augmentation)
        if train:
            X.append(np.hstack([emb[0], emb[2], emb[1]]))
            y.append(0)
    X = np.vstack(X)
    y = np.hstack(y)
    return X, y


# Hint: adjust batch_size and num_workers to your PC configuration, so that you
# don't run out of memory (VRAM if on GPU, RAM if on CPU)
def create_loader_from_np(X, y=None, train=True, batch_size=global_batch_size, shuffle=True, num_workers=global_num_workers):
    """
    Create a torch.utils.data.DataLoader object from numpy arrays containing the data.

    input: X: numpy array, the features
           y: numpy array, the labels

    output: loader: torch.data.util.DataLoader, the object containing the data
    """
    if train:
        # Attention: If you get type errors you can modify the type of the
        # labels here
        dataset = TensorDataset(torch.from_numpy(X).type(torch.float),
                                torch.from_numpy(y).type(torch.long))
    else:
        dataset = TensorDataset(torch.from_numpy(X).type(torch.float))

    loader = DataLoader(dataset=dataset,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        pin_memory=True, num_workers=num_workers)
    return loader


# TODO: define a model. Here, the basic structure is defined, but you need to fill in the details
class Net(nn.Module):
    """
    The model class, which defines our classifier.
    """

    def __init__(self):
        """
        The constructor of the model.
        """
        super().__init__()
        self.fc = nn.Linear(1536, 512)  # where does the 1536 come from? 512*3, yes, but why *3 ?
        self.out = nn.Linear(512, 1, bias=True)  # where does the 1536 come from? 512*3, yes, but why *3 ?

    def forward(self, x):
        """
        The forward pass of the model.

        input: x: torch.Tensor, the input to the model

        output: x: torch.Tensor, the output of the model
        """
        x = F.relu(self.fc(x))
        x = self.out(x)
        return x


def train_model(train_loader):
    """
    The training procedure of the model; it accepts the training data, defines the model 
    and then trains it.

    input: train_loader: torch.data.util.DataLoader, the object containing the training data

    output: model: torch.nn.Module, the trained model
    """
    model = Net()
    model.train()
    model.to(device)
    n_epochs = 10

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # TODO: define a loss function, optimizer and proceed with training. Hint: use the part
    # of the training data as a validation split. After each epoch, compute the loss on the
    # validation split and print it out. This enables you to see how your model is performing
    # on the validation data before submitting the results on the server. After choosing the
    # best model, train it on the whole training data.
    if testing:
        best_model_params_path = 'dataset/best'
        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0
        for epoch in range(n_epochs):
            if verbose:
                print(f'Epoch {epoch}/{n_epochs - 1}')
                print('-' * 20)
            generator = torch.Generator().manual_seed(42)
            train_split = torch.utils.data.random_split(train_loader.dataset, [0.2, 0.2, 0.6], generator=generator)

            for [X, y] in DataLoader(train_split[2], shuffle=True, batch_size=global_batch_size):
                inputs = X.to(device)
                labels = y.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(True):
                    outputs = model(inputs)
                    loss = criterion(outputs.squeeze(), labels.float())

                    # backward + optimize only if in training phase
                    loss.backward()
                    optimizer.step()

            running_loss = 0.0
            running_corrects = 0

            for [X, y] in DataLoader(train_split[1], shuffle=False, batch_size=global_batch_size):
                inputs = X.to(device)
                labels = y.to(device)

                with torch.set_grad_enabled(False):
                    predicted = model(X)
                    predicted = predicted.cpu().numpy()
                    # Rounding the predictions to 0 or 1
                    predicted[predicted >= 0.5] = 1
                    predicted[predicted < 0.5] = 0

                running_loss += loss.item() * inputs.size(0)
                # running_corrects += global_batch_size - np.sum(np.abs(predicted - torch.Tensor.numpy(y)))

            epoch_loss = running_loss / len(train_split[1].indices)
            # epoch_acc = running_corrects / len(train_split[1].indices)

            print(f'Loss: {epoch_loss:.4f}')  # Acc: {epoch_acc:.4f}')

            # deep copy the model
#           if epoch_acc > best_acc:
#               best_acc = epoch_acc
#               torch.save(model.state_dict(), best_model_params_path)

#       model.load_state_dict(torch.load(best_model_params_path))

    else:
        for epoch in range(n_epochs):
            if verbose:
                print(f'Epoch {epoch}/{n_epochs - 1}')
                print('-' * 20)

            running_loss = 0.0

            for [X, y] in train_loader:
                inputs = X.to(device)
                labels = y.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(True):
                    outputs = model(inputs)

                    loss = criterion(outputs.squeeze(), labels.float())

                    # backward + optimize only if in training phase
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(train_loader.dataset)
            print(f'Loss: {epoch_loss:.4f}')  # Acc: {epoch_acc:.4f}')

    return model


def test_model(model, loader):
    """
    The testing procedure of the model; it accepts the testing data and the trained model and
    then tests the model on it.

    input: model: torch.nn.Module, the trained model
           loader: torch.data.util.DataLoader, the object containing the testing data

    output: None, the function saves the predictions to a results.txt file
    """
    model.eval()
    predictions = []
    # Iterate over the test data
    with torch.no_grad():   # We don't need to compute gradients for testing
        for [x_batch] in loader:
            x_batch = x_batch.to(device)
            predicted = model(x_batch)
            predicted = predicted.cpu().numpy()
            # Rounding the predictions to 0 or 1
            predicted[predicted >= 0.5] = 1
            predicted[predicted < 0.5] = 0
            predictions.append(predicted)
        predictions = np.vstack(predictions)
    np.savetxt("results.txt", predictions, fmt='%i')


# Main function. You don't have to change this
if __name__ == '__main__':
    TRAIN_TRIPLETS = 'train_triplets.txt'
    TEST_TRIPLETS = 'test_triplets.txt'

    # generate embedding for each image in the dataset
    if (os.path.exists('dataset/embeddings.npy') is False):
        generate_embeddings()

    # load the training data
    X, y = get_data(TRAIN_TRIPLETS)
    # Create data loaders for the training data
    train_loader = create_loader_from_np(X, y, train=True, batch_size=global_batch_size)
    # delete the loaded training data to save memory, as the data loader copies
    del X
    del y

    # repeat for testing data
    X_test, y_test = get_data(TEST_TRIPLETS, train=False)
    test_loader = create_loader_from_np(X_test, train=False, batch_size=2048, shuffle=False)
    del X_test
    del y_test

    # define a model and train it
    model = train_model(train_loader)

    # test the model on the test data
    test_model(model, test_loader)
    print("Results saved to results.txt")
