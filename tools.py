import matplotlib.pyplot as plt
import numpy as np


def visualize(train, true, pred):
    plt.figure(figsize=(15, 3), dpi=300)
    plt.grid(True, color='#DBDBDB', zorder=0)
    plt.plot(np.concatenate([train, true]), label='True')
    plt.plot(np.concatenate([train, pred]), label='Pred')
    plt.plot(train, color='black')
    plt.xticks(np.linspace(0, 12 * 12, 13))
    plt.legend()
