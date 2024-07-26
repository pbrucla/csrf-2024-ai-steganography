import matplotlib.pyplot as plt

def test_plot_data():
    epoch_array = [0, 1, 2, 3, 4]
    loss_values = [0.6, 0.5, 0.4, 0.3, 0.2]
    accu_values = [50, 60, 70, 80, 90]
    
    print("Starting plot_data graph")
    
    plt.subplot(2,1,1)
    plt.plot(epoch_array, loss_values, label = "Loss over the Epochs", color = "orange")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    
    plt.subplot(2,1,2)
    plt.plot(epoch_array, accu_values, label = "Accuracy over the Epochs", color = "green")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    # Adjust subplot to give some padding
    plt.tight_layout()
    plt.show() 

    print("plot_data graph finished")

if __name__ == "__main__":
    test_plot_data()