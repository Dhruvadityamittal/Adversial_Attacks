
import matplotlib.pylab as plt
import torch 

def plot_orignal(train_mnist):
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    imgs,labels = next(iter(train_mnist))


    for i in range(1,cols*rows +1):
        sample_idx = torch.randint(len(imgs), size=(1,)).item() 
        figure.add_subplot(rows,cols,i)
        plt.imshow(imgs[sample_idx][0],cmap="gray")
        plt.title(labels[sample_idx].item())   
        plt.axis("off")
    plt.suptitle("Orignal Data")
    plt.savefig("Orignal_data.png")
    plt.show()

def plot_pertubations(epsilons,examples):

    cnt = 0
    plt.figure(figsize=(8,10))
    plt.suptitle("Pertubated Data")
    for i in range(len(epsilons)):
        for j in range(len(examples[i])):
            cnt += 1
            plt.subplot(len(epsilons),len(examples[0]),cnt)
            plt.xticks([], [])
            plt.yticks([], [])
            if j == 0:
                plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=14)
            orig,adv,ex = examples[i][j]
            plt.title("{} -> {}".format(orig, adv))
            plt.imshow(ex, cmap="gray")
    plt.tight_layout()

    plt.savefig("Pertubated_Data.png")
    plt.show()
