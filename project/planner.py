import torch
import torch.nn.functional as F


def spatial_argmax(logit):
    """
    Compute the soft-argmax of a heatmap
    :param logit: A tensor of size BS x H x W
    :return: A tensor of size BS x 2 the soft-argmax in normalized coordinates (-1 .. 1)
    Note: BS stands for batch size
    """
    weights = F.softmax(logit.view(logit.size(0), -1), dim=-1).view_as(logit)
    return torch.stack(((weights.sum(1) * torch.linspace(-1, 1, logit.size(2)).to(logit.device)[None]).sum(1),
                        (weights.sum(2) * torch.linspace(-1, 1, logit.size(1)).to(logit.device)[None]).sum(1)), 1)


class Planner(torch.nn.Module):
    def __init__(self):
        super().__init__()

        """
        Your code here. Fill in `torch.nn.Sequential()` with your network layers.
        Tools/tips for building your network:
              * Look at https://pytorch.org/docs/stable/nn.html for reference on the different
                building blocks PyTorch gives you.
              * This project implements a Convolutional Neural Network (CNN), so you'll want to
                look for the module corresponding to a convolutional layer
                    - Filter sizes are generallly 3x3, 5x5, 7x7
              * Any time you add a convolutional layer, follow it with a ReLU layer
              * You'll want to use the "2d" variants of the different modules/layers
              
        Additional Tips
              * Create a "unit" of 2-3 different layers, and repeat that a few (no more than three) times
                throughout your network, varying the input/output dimensions each time
              * You generally want to increase the # of channels as you move through the network
              # Google common CNN architectures and see how you might be able to replicate them with PyTorch.
        """
        self.network = torch.nn.Sequential(
          torch.nn.BatchNorm2d(3),
          torch.nn.Conv2d(3, 16, 5, 2, 2),
          torch.nn.ReLU(True),
          torch.nn.BatchNorm2d(16),
          torch.nn.Conv2d(16, 32, 5, 2, 2),
          torch.nn.ReLU(True),
          torch.nn.BatchNorm2d(32),
          torch.nn.Conv2d(32, 32, 5, 2, 2),
          torch.nn.ReLU(True),
          torch.nn.BatchNorm2d(32),
          torch.nn.Conv2d(32, 32, 5, 2, 2),
          torch.nn.ReLU(True),
          # Keep this as the last layer, but feel free to modify any of the 
          # layers above!
          torch.nn.Conv2d(32, 1, 1)
        )

    def forward(self, img):
        """
        Your code here
        Predict the aim point in image coordinate, given the supertuxkart image
        @img: (BS,3,96,128)
        return (BS,2)
        Note: BS stands for batch size, i.e. the number of images being processed
              3 is the number of channels in the image (the RGB channels since it is a color image)
              96 and 128 are the height and width of each image (96x128)
        """
        # We run the input image through the network we created above.
        x = self.network(img)
        
        # We take the output of our network and run it through a final activation function
        # to give us probabilities for different possible aim points
        return spatial_argmax(x[:, 0])


def save_model(model):
    from torch import save
    from os import path
    if isinstance(model, Planner):
        return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'planner.th'))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model():
    from torch import load
    from os import path
    r = Planner()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'planner.th'), map_location='cpu'))
    return r

def test_planner(pytux, track, verbose):
    from .controller import control
    # Load model
    planner = load_model().eval()
    for t in track:
        steps, how_far = pytux.rollout(t, control, planner=planner, max_frames=1000, verbose=verbose)
        print(steps, how_far)
    pytux.close()


if __name__ == '__main__':
    from .controller import control
    from .utils import PyTux
    from argparse import ArgumentParser


    def test_planner(args):
        # Load model
        planner = load_model().eval()
        pytux = PyTux()
        for t in args.track:
            steps, how_far = pytux.rollout(t, control, planner=planner, max_frames=1000, verbose=args.verbose)
            print(steps, how_far)
        pytux.close()


    parser = ArgumentParser("Test the planner")
    parser.add_argument('track', nargs='+')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    test_planner(args)
