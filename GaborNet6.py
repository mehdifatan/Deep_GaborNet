import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import os


class GaborLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(GaborLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Initialize the Gabor filters as trainable parameters
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize weights with Gabor filters for simplicity, real applications might need trainable parameters
        self.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x):
        return F.conv2d(x, self.weight, None, self.stride, self.padding)

class TrainableGaborFilter(nn.Module):
    def __init__(self, size, in_channels=3):
        super(TrainableGaborFilter, self).__init__()
        self.size = size
        self.in_channels = in_channels
        
        # Initialize parameters
		
		# lambda represents the wavelength of the sinusoidal factor, 
		# theta represents the orientation of the normal to the parallel stripes of a Gabor function, 
		# psi is the phase offset, 
		# sigma is the sigma/standard deviation of the Gaussian envelope and 
		# gamma is the spatial aspect ratio, and specifies the ellipticity of the support of the Gabor function.
		
        self.sigma = nn.Parameter(torch.rand(1) * 10 + 1)  # Sigma in range [1, 11]
        self.theta = nn.Parameter(torch.rand(1) * np.pi)  # Theta in range [0, pi]
        self.lamda = nn.Parameter(torch.rand(1) * 20 + 10)  # Lambda in range [10, 30]
        self.gamma = nn.Parameter(torch.rand(1) * 0.5 + 0.5)  # Gamma in range [0.5, 1]
        self.psi = nn.Parameter(torch.rand(1) * 2 * np.pi - np.pi)  # Psi in range [-pi, pi]

    def forward(self, x):
        # Generate the Gabor kernel based on current parameters
        kernel = self._gabor_kernel(self.size, self.sigma, self.theta, self.lamda, self.gamma, self.psi)
        # Repeat kernel for each input channel
        kernel = kernel.repeat(self.in_channels, 1, 1, 1)
        # Apply the Gabor kernel to the input
        return F.conv2d(x, kernel, padding=self.size//2, groups=self.in_channels)

    def _gabor_kernel(self, size, sigma, theta, lamda, gamma, psi):
        # Implementation of Gabor kernel generation
        sigma_x = sigma
        sigma_y = sigma / gamma

        xmax = size // 2
        ymax = size // 2
        xmin = -xmax
        ymin = -ymax
        x, y = torch.meshgrid(torch.linspace(xmin, xmax, size), torch.linspace(ymin, ymax, size))
        x = x.to(sigma.device)
        y = y.to(sigma.device)

        x_theta = x * torch.cos(theta) + y * torch.sin(theta)
        y_theta = -x * torch.sin(theta) + y * torch.cos(theta)

        gb = torch.exp(-.5 * (x_theta**2 / sigma_x**2 + y_theta**2 / sigma_y**2)) * torch.cos(2 * np.pi / lamda * x_theta + psi)
        
        return gb.view(1, 1, size, size)




class TrainableGaborFilter2(nn.Module):
    def __init__(self, size, in_channels=3, out_channels=6):
        super(TrainableGaborFilter2, self).__init__()
        self.size = size
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Initialize parameters for each output channel
        self.sigma = nn.Parameter(torch.rand(out_channels) * 10 + 1)  # Sigma in range [1, 11]
        self.theta = nn.Parameter(torch.rand(out_channels) * np.pi)  # Theta in range [0, pi]
        self.lamda = nn.Parameter(torch.rand(out_channels) * 20 + 10)  # Lambda in range [10, 30]
        self.gamma = nn.Parameter(torch.rand(out_channels) * 0.5 + 0.5)  # Gamma in range [0.5, 1]
        self.psi = nn.Parameter(torch.rand(out_channels) * 2 * np.pi - np.pi)  # Psi in range [-pi, pi]

    def forward(self, x):
        # Generate and stack Gabor kernels for each output channel
        kernels = [self._gabor_kernel(self.size, self.sigma[i], self.theta[i], 
                                      self.lamda[i], self.gamma[i], self.psi[i]).unsqueeze(0)
                   for i in range(self.out_channels)]
        kernel = torch.cat(kernels, dim=0)
        # Adjust kernel shape for convolution
        kernel = kernel.repeat(1, self.in_channels, 1, 1)
        return F.conv2d(x, kernel, padding=self.size//2)

    def _gabor_kernel(self, size, sigma, theta, lamda, gamma, psi):
        # Implementation of Gabor kernel generation
        sigma_x = sigma
        sigma_y = sigma / gamma
        xmax = size // 2
        ymax = size // 2
        xmin = -xmax
        ymin = -ymax
        x, y = torch.meshgrid(torch.linspace(xmin, xmax, size), torch.linspace(ymin, ymax, size))
        x = x.to(sigma.device)
        y = y.to(sigma.device)

        x_theta = x * torch.cos(theta) + y * torch.sin(theta)
        y_theta = -x * torch.sin(theta) + y * torch.cos(theta)

        gb = torch.exp(-.5 * (x_theta**2 / sigma_x**2 + y_theta**2 / sigma_y**2)) * torch.cos(2 * np.pi / lamda * x_theta + psi)
        return gb.view(1, size, size)        
        


# Define a trainable 1D Gabor filter
class TrainableGaborFilter1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(TrainableGaborFilter1D, self).__init__()
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Gabor parameters: wavelength, orientation, phase offset, bandwidth
        self.lambda_ = nn.Parameter(torch.rand(out_channels, in_channels) * 10 + 1)
        self.theta = nn.Parameter(torch.rand(out_channels, in_channels) * np.pi)
        self.psi = nn.Parameter(torch.rand(out_channels, in_channels) * np.pi)
        self.sigma = nn.Parameter(torch.rand(out_channels, in_channels) * 5 + 1)
        self.gamma = nn.Parameter(torch.ones(out_channels, in_channels))

    def forward(self, x):
        # Generate the Gabor filter dynamically
        filters = torch.zeros((self.out_channels, self.in_channels, self.kernel_size), device=x.device)
        for i in range(self.out_channels):
            for j in range(self.in_channels):
                t = torch.arange(self.kernel_size, device=x.device) - self.kernel_size // 2
                theta = self.theta[i, j]
                sigma = self.sigma[i, j]
                lambda_ = self.lambda_[i, j]
                psi = self.psi[i, j]
                gamma = self.gamma[i, j]
                
                # 1D Gabor equation
                gabor = torch.exp(-0.5 * ((t ** 2) / sigma**2)) * torch.cos(2 * np.pi * t / lambda_ + psi)
                filters[i, j] = gabor

        return F.conv1d(x, filters, stride=1, padding=self.kernel_size // 2)
    
    
    
    

class GaborNet(nn.Module):
    def __init__(self):
        super(GaborNet, self).__init__()
        self.gabor = TrainableGaborFilter2(size=11, in_channels=1, out_channels=11)
        self.gabor2 = TrainableGaborFilter2(size=11, in_channels=11, out_channels=11)
        self.gabor3 = TrainableGaborFilter2(size=11, in_channels=11, out_channels=11)
        self.gabor4 = TrainableGaborFilter2(size=11, in_channels=11, out_channels=11)
        self.gabor5 = TrainableGaborFilter2(size=11, in_channels=11, out_channels=11)
        self.gabor6 = TrainableGaborFilter2(size=11, in_channels=11, out_channels=11)
        self.gabor7 = TrainableGaborFilter2(size=11, in_channels=11, out_channels=11)
        self.gabor8 = TrainableGaborFilter2(size=11, in_channels=11, out_channels=11)
        self.gabor9 = TrainableGaborFilter2(size=11, in_channels=11, out_channels=11)
        self.conv2 = nn.Conv2d(12, 16, kernel_size=2, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.fc1 = nn.Linear(784, 120)  # Adjusted size for CIFAR-10
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.mfc = nn.Linear(3179, 5)
        # Replace fully connected layers with 1D Gabor filters
        #self.gabor1d_1 = TrainableGaborFilter1D(in_channels=1, out_channels=10, kernel_size=21)
        #self.gabor1d_2 = TrainableGaborFilter1D(in_channels=10, out_channels=5, kernel_size=21)
        #self.gabor1d_2 = TrainableGaborFilter1D(in_channels=10, out_channels=10, kernel_size=21)
        
        
    def forward(self, x):
        x = F.relu(self.gabor(x))
        x = F.relu(self.gabor2(x))
        x = F.relu(self.gabor3(x))
        x = F.relu(self.gabor4(x))
        #x = F.relu(self.gabor5(x))
        #x = F.relu(self.gabor6(x))
        #x = F.relu(self.gabor7(x))
        #x = F.relu(self.gabor8(x))
        #x = F.relu(self.gabor9(x))
        #x = self.pool(x)
        #x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        #x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        #x = self.fc3(x)
        
        # Flatten and reshape for 1D Gabor filters
        #x = torch.flatten(x, 1).unsqueeze(1)  # Add channel dimension for 1D conv
        #x = F.relu(self.gabor1d_1(x))
        #x = self.gabor1d_2(x)

        #return x.squeeze()
        
        x = self.mfc(x)
        return x



# Training Function
def train(net, trainloader, epochs=5):
    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=0.1)
    
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 200 == 199:
                print(f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 2000}')
                running_loss = 0.0

    print('Finished Training')



# Testing Function
def test(net, testloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total} %')



# Visualization of Layer Activations
def visualize_layer_activations(net, loader):
    images, _ = next(iter(loader))
    images = images[:4]  # Take the first 4 images from the batch

    activations = []
    def hook_fn(module, input, output):
        activations.append(output)

    hooks = []
    for layer in net.children():
        hooks.append(layer.register_forward_hook(hook_fn))
    
    _ = net(images)

    for i, activation in enumerate(activations):
        act = activation[0].detach().cpu()
        num_subplots = act.size(0)
        
        fig, axes = plt.subplots(1, min(num_subplots, 4), figsize=(num_subplots*2.5, 2.5))
        fig.suptitle(f'Layer {i+1} Activations')
        
        for j in range(min(num_subplots, 4)):
            ax = axes[j] if num_subplots > 1 else axes
            ax.imshow(act[j], cmap='viridis')
            ax.axis('off')
        
        plt.show()

    for hook in hooks:
        hook.remove()



# def visualize_gabor_filters(layer):
    # with torch.no_grad():
        # weights = layer.weight.cpu().numpy()
        # filter_count = weights.shape[0]
        
        # Set the number of filter images in a row
        # row_size = min(8, filter_count)  # Change 8 to another number if you want more columns
        # fig, axs = plt.subplots((filter_count // row_size) + (0 if filter_count % row_size == 0 else 1), row_size, figsize=(row_size * 3, (filter_count // row_size) * 3))
        # fig.subplots_adjust(hspace = .5, wspace=.001)
        # axs = axs.ravel()
        
        # for i in range(filter_count):
            # axs[i].imshow(weights[i, 0]*100, cmap='gray')  # Assuming single channel for simplicity
            # axs[i].set_title(f'Filter {i+1}')
            # axs[i].axis('off')
        
        # plt.show()



def visualize_gabor_filters(layer):
    with torch.no_grad():
        weights = layer.weight.cpu().numpy()  # Assuming the weights are 4D: [out_channels, in_channels, H, W]
        filter_count = weights.shape[0]
        
        # Calculate grid size for subplots
        row_size = min(8, filter_count)  # Number of columns
        col_size = (filter_count // row_size) + (0 if filter_count % row_size == 0 else 1)  # Number of rows
        
        fig = plt.figure(figsize=(row_size * 3, col_size * 3))
        
        # Generate a meshgrid for filter dimensions
        x = np.linspace(0, weights.shape[2] - 1, weights.shape[2])
        y = np.linspace(0, weights.shape[3] - 1, weights.shape[3])
        x, y = np.meshgrid(x, y)
        
        for i in range(filter_count):
            ax = fig.add_subplot(col_size, row_size, i + 1, projection='3d')
            # Normalize weights to enhance visibility, assuming single channel for simplicity
            weight = weights[i, 0]
            z = weight
            # Plot surface
            surf = ax.plot_surface(x, y, z, cmap='viridis', edgecolor='none')
            ax.set_zlim(weight.min(), weight.max())
            ax.set_title(f'Filter {i+1}')
            ax.axis('off')
        
        plt.tight_layout()
        plt.show()




def visualize_gabor_filters_comparison(initial_layer, trained_layer, title_prefix=''):
    with torch.no_grad():
        initial_weights = initial_layer.weight.cpu().numpy()
        trained_weights = trained_layer.weight.cpu().numpy()
        out_channels = initial_weights.shape[0]
        
        fig = plt.figure(figsize=(20, 12))
        
        for i in range(min(out_channels, 12)):  # Visualize the first 6 filters for simplicity
            # Initial filters
            ax = fig.add_subplot(2, 6, i + 1, projection='3d')
            x, y = np.meshgrid(range(initial_weights.shape[2]), range(initial_weights.shape[3]))
            z = initial_weights[i, 0]
            ax.plot_surface(x, y, z, cmap='viridis')
            ax.set_title(f'{title_prefix} Initial Filter {i+1}')
            ax.set_zlim([np.min(initial_weights), np.max(initial_weights)])
            
            # Trained filters
            ax = fig.add_subplot(2, 6, i + 7, projection='3d')  # Offset by 6 for the second row
            z = trained_weights[i, 0]
            ax.plot_surface(x, y, z, cmap='viridis')
            ax.set_title(f'{title_prefix} Trained Filter {i+1}')
            ax.set_zlim([np.min(trained_weights), np.max(trained_weights)])
        
        plt.tight_layout()
        plt.show()




def visualize_gabor_filters_comparison_top_view(initial_layer, trained_layer, title_prefix=''):
    with torch.no_grad():
    
        # Generate Gabor kernels from the initial and trained layers
        initial_kernel = initial_layer._gabor_kernel(
            initial_layer.size, initial_layer.sigma, initial_layer.theta, initial_layer.lamda, initial_layer.gamma, initial_layer.psi
        ).squeeze().cpu().numpy()

        trained_kernel = trained_layer._gabor_kernel(
            trained_layer.size, trained_layer.sigma, trained_layer.theta, trained_layer.lamda, trained_layer.gamma, trained_layer.psi
        ).squeeze().cpu().numpy()


        # Setup for 3D plot
        x, y = np.meshgrid(range(initial_kernel.shape[1]), range(initial_kernel.shape[0]))
        fig = plt.figure(figsize=(12, 6))
        
        
        #print(initial_kernel.shape)
        
        
        # Initial kernel visualization
        ax1 = fig.add_subplot(2, 1, 1, projection='3d')
        ax1.plot_surface(x, y, initial_kernel,cmap='viridis')
        ax1.view_init(elev=90, azim=-90)  # Top-down view
        ax1.set_title(f'{title_prefix} Initial Gabor Kernel')
        ax1.set_zlim([np.min(initial_kernel), np.max(initial_kernel)])
        ax1.axis('off')  # Hide axes for clarity
        
        # Trained kernel visualization
        ax2 = fig.add_subplot(2, 1, 2, projection='3d')
        ax2.plot_surface(x, y, trained_kernel, cmap='viridis')
        ax2.view_init(elev=90, azim=-90)  # Top-down view
        ax2.set_title(f'{title_prefix} Trained Gabor Kernel')
        ax2.set_zlim([np.min(trained_kernel), np.max(trained_kernel)])
        ax2.axis('off')  # Hide axes for clarity
        
        plt.tight_layout()
        plt.show()




def visualize_gabor_filters_comparison_top_view2(initial_layer, trained_layer, title_prefix=''):
    
    
    with torch.no_grad():
        
        
        
        kernels = [initial_layer._gabor_kernel(initial_layer.size, initial_layer.sigma[i], initial_layer.theta[i], 
                                      initial_layer.lamda[i], initial_layer.gamma[i], initial_layer.psi[i]).unsqueeze(0)
                   for i in range(initial_layer.out_channels)]
        initial_kernel = torch.cat(kernels, dim=0)
        
        
        kernels = [trained_layer._gabor_kernel(trained_layer.size, trained_layer.sigma[i], trained_layer.theta[i], 
                                      trained_layer.lamda[i], trained_layer.gamma[i], trained_layer.psi[i]).unsqueeze(0)
                   for i in range(trained_layer.out_channels)]
        trained_kernel = torch.cat(kernels, dim=0)
        
        
        
        #initial_kernel= initial_layer._gabor_kernel(
        #    initial_layer.size, initial_layer.sigma, initial_layer.theta, initial_layer.lamda, initial_layer.gamma, initial_layer.psi
        #).squeeze().cpu().numpy()

        
        #trained_kernel= trained_layer._gabor_kernel(
        #    trained_layer.size, trained_layer.sigma, trained_layer.theta, trained_layer.lamda, trained_layer.gamma, trained_layer.psi
        #).squeeze().cpu().numpy()
    
        out_channels = initial_kernel.shape[0]
        #rint(out_channels)


        
        print(initial_kernel.shape)
        
        fig = plt.figure(figsize=(20, 6))
        
        for i in range(min(out_channels, 6)):  # Visualize the first 6 filters for simplicity


            #initial_kernel= initial_layer[i, 0]._gabor_kernel(
            #initial_layer.size, initial_layer.sigma, initial_layer.theta, initial_layer.lamda, initial_layer.gamma, initial_layer.psi
            #).squeeze().cpu().numpy()
        
            #trained_kernel= trained_layer[i, 0]._gabor_kernel(
            #trained_layer.size, trained_layer.sigma, trained_layer.theta, trained_layer.lamda, trained_layer.gamma, trained_layer.psi
            #).squeeze().cpu().numpy()
    



            # Initial filters
            ax_init = fig.add_subplot(2, 6, i + 1, projection='3d')
            x, y = np.meshgrid(range(initial_kernel.shape[1]), range(initial_kernel.shape[0]))
            z = initial_kernel[i, 0]*2
            ax_init.plot_surface(x, y, z,cmap='viridis')
            ax_init.view_init(elev=90, azim=-90)  # Top-down view
            ax_init.set_title(f'{title_prefix} Initial Filter {i+1}')
            ax_init.set_zlim([np.min(initial_kernel), np.max(initial_kernel)])
            ax_init.axis('off')  # Hide axes for clarity
            
            # Trained filters
            ax_trained = fig.add_subplot(2, 6, i + 7, projection='3d')
            z = trained_kernel[i, 0]*2
            x, y = np.meshgrid(range(trained_kernel.shape[1]), range(trained_kernel.shape[0]))
            ax_trained.plot_surface(x, y, z, cmap='viridis')
            ax_trained.view_init(elev=90, azim=-90)  # Top-down view
            ax_trained.set_title(f'{title_prefix} Trained Filter {i+1}')
            ax_trained.set_zlim([np.min(trained_kernel), np.max(trained_kernel)])
            ax_trained.axis('off')  # Hide axes for clarity
        
        plt.tight_layout()
        plt.show()





def visualize_gabor_filters_comparison_top_view3(initial_kernel, trained_kernel, title_prefix=''):
    with torch.no_grad():
        # Assuming the rest of the function setup remains unchanged...

        fig = plt.figure(figsize=(20, 6))

        for i in range(min(initial_kernel.out_channels, 6)):  # Visualize the first 6 filters
            # Correct meshgrid generation to match the filter size
            x, y = np.meshgrid(np.arange(initial_kernel.size), np.arange(initial_kernel.size))

            # Select the i-th filter from initial and trained layers
            # Ensure z arrays are 2D matching the size of the Gabor filter
            z_initial = initial_kernel[i, 0, :, :]  # Assuming initial_kernel is [out_channels, 1, H, W]
            z_trained = trained_kernel[i, 0, :, :]  # Assuming trained_kernel is [out_channels, 1, H, W]

            # Initial filters visualization
            ax_init = fig.add_subplot(2, 6, i + 1, projection='3d')
            ax_init.plot_surface(x, y, z_initial, cmap='viridis')
            ax_init.view_init(elev=90, azim=-90)
            ax_init.set_title(f'{title_prefix} Initial Filter {i+1}')
            ax_init.axis('off')

            # Trained filters visualization
            ax_trained = fig.add_subplot(2, 6, i + 7, projection='3d')
            ax_trained.plot_surface(x, y, z_trained, cmap='viridis')
            ax_trained.view_init(elev=90, azim=-90)
            ax_trained.set_title(f'{title_prefix} Trained Filter {i+1}')
            ax_trained.axis('off')

        plt.tight_layout()
        plt.show()





def visualize_gabor_filters_comparison_top_view4(initial_layer, trained_layer, title_prefix=''):
    with torch.no_grad():
    
        fig = plt.figure(figsize=(20, 6))
        
        for i in range(6):

            # Generate Gabor kernels from the initial and trained layers
            initial_kernel = initial_layer._gabor_kernel(
                initial_layer.size, initial_layer.sigma[i], initial_layer.theta[i], initial_layer.lamda[i], initial_layer.gamma[i], initial_layer.psi[i]
            ).squeeze().cpu().numpy()

            trained_kernel = trained_layer._gabor_kernel(
                trained_layer.size, trained_layer.sigma[i], trained_layer.theta[i], trained_layer.lamda[i], trained_layer.gamma[i], trained_layer.psi[i]
            ).squeeze().cpu().numpy()


            # Setup for 3D plot
            x, y = np.meshgrid(range(initial_kernel.shape[1]), range(initial_kernel.shape[0]))
            
            
            
            #print(initial_kernel.shape)
            
            
            # Initial kernel visualization
            ax1 = fig.add_subplot(2, 6, i+1, projection='3d')
            ax1.plot_surface(x, y, initial_kernel,cmap='viridis')
            ax1.view_init(elev=90, azim=-90)  # Top-down view
            ax1.set_title(f'{title_prefix} Initial Gabor Kernel '+str(i+1))
            ax1.set_zlim([np.min(initial_kernel), np.max(initial_kernel)])
            ax1.axis('off')  # Hide axes for clarity
            
            # Trained kernel visualization
            ax2 = fig.add_subplot(2, 6, i+7, projection='3d')
            ax2.plot_surface(x, y, trained_kernel, cmap='viridis')
            ax2.view_init(elev=90, azim=-90)  # Top-down view
            ax2.set_title(f'{title_prefix} Trained Gabor Kernel '+str(i+1))
            ax2.set_zlim([np.min(trained_kernel), np.max(trained_kernel)])
            ax2.axis('off')  # Hide axes for clarity
            
        plt.tight_layout()
        plt.show()





# Main
if __name__ == '__main__':
    
    
    
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    # # Data Loading
    # transform = transforms.Compose(
    #     #[transforms.ToTensor(),
    #     # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
         
    #     [transforms.ToTensor(),
    #      transforms.Normalize((1, 1, 1), (1, 1, 1))])
    
    
    batch_size=16
    epn=100
    
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(),  # Convert images to grayscale
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize images
    ])



    # Define transformations
    # # transform = transforms.Compose([
        # transforms.RandomResizedCrop(224),
        # transforms.RandomHorizontalFlip(),
        # # transforms.ToTensor(),
        # # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # # ])


    # Load the dataset
    data_dir = r'C:\Users\mfatan\Desktop\visnet\sierpinski_triangles_dataset3\sierpinski_triangles_dataset3'
    full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    # Split the dataset into training and validation sets
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    # Create dataloaders
    #batch_size = 32

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

         

    # trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
    #                                         download=True, transform=transform)
        
    # trainloader = DataLoader(trainset, batch_size=16,
    #                          shuffle=True, num_workers=2)



    # testset = torchvision.datasets.CIFAR10(root='./data', train=False,
    #                                        download=True, transform=transform)
                                           
    # testloader = DataLoader(testset, batch_size=16,
    #                         shuffle=False, num_workers=2)
                            
                            
                            
    net = GaborNet()
    initial_net=copy.deepcopy(net)
    train(net, trainloader, epochs=epn)  # Reduced epochs for quick testing
    test(net, testloader)
    #visualize_gabor_filters(net2.gabor)
    #visualize_gabor_filters(net.gabor)
    #visualize_gabor_filters(layer)
    #visualize_layer_activations(net, trainloader)
    # Visualize Gabor filters before and after training in one plot
    #visualize_gabor_filters_comparison(initial_net.gabor, net.gabor, title_prefix='Gabor')
    visualize_gabor_filters_comparison_top_view4(initial_net.gabor, net.gabor, title_prefix='Layer1')
    visualize_gabor_filters_comparison_top_view4(initial_net.gabor2, net.gabor2, title_prefix='Layer2')
    visualize_gabor_filters_comparison_top_view4(initial_net.gabor3, net.gabor3, title_prefix='Layer')
    visualize_gabor_filters_comparison_top_view4(initial_net.gabor4, net.gabor4, title_prefix='Layer4')
    visualize_gabor_filters_comparison_top_view4(initial_net.gabor5, net.gabor5, title_prefix='Layer5')
    #visualize_gabor_filters_comparison_top_view4(initial_net.gabor6, net.gabor6, title_prefix='Layer6')
    #visualize_gabor_filters_comparison_top_view4(initial_net.gabor7, net.gabor7, title_prefix='Layer7')
    #visualize_gabor_filters_comparison_top_view4(initial_net.gabor8, net.gabor8, title_prefix='Layer8')
    #visualize_gabor_filters_comparison_top_view4(initial_net.gabor9, net.gabor9, title_prefix='Layer9')
    