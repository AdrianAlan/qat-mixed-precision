import matplotlib.pyplot as plt


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt

    def reset(self):
        self.count = 0
        self.average = 0
        self.sum = 0

    def update(self, value, num=1):
        self.count += num
        self.sum += value * num
        self.average = self.sum / self.count

    def __str__(self):
        fmtstr = '{average' + self.fmt + '} ({name})'
        return fmtstr.format(**self.__dict__)


class Plotting():
    """Plots the training and test loss and accuracy per epoch"""

    def __init__(self, fname):
        self.fname = fname

    def draw(self, train_loss, train_accuracy, test_loss, test_accuracy):
        fig, axs = plt.subplots(2, 2, figsize=(25, 20))

        axs[0, 0].set_title('Train Loss')
        axs[0, 1].set_title('Training Accuracy')
        axs[1, 0].set_title('Test Loss')
        axs[1, 1].set_title('Test Accuracy')

        axs[0, 0].plot(train_loss)
        axs[0, 1].plot(train_accuracy)
        axs[1, 0].plot(test_loss)
        axs[1, 1].plot(test_accuracy)

        fig.savefig(self.fname)
