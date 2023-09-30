import matplotlib.pyplot as plt


def show_image(image):
    plt.imshow(image.permute(1, 2, 0))
    plt.axis(False)
    plt.show()
