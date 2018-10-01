from keras.layers import Input, Dense, Lambda, Layer
from keras.models import Model, load_model
from keras import backend as K
from keras import metrics
from keras.datasets import mnist
from keras.utils import to_categorical

import matplotlib.pyplot as plt

original_dim = 28 * 28
latent_dim = 2
intermediate_dim = 74
batch_size = 100
epsilon_std = 1.0
epochs = 50

(x_train, _), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

x_train = x_train.reshape((-1, original_dim))
x_test = x_test.reshape((-1, original_dim))

def train():
    x = Input(shape=(original_dim,))

    encoder = Dense(intermediate_dim, activation='relu')(x)
    z_mean = Dense(latent_dim)(encoder)
    z_log_var = Dense(latent_dim)(encoder)

    def sample(input_args):
        z_mean, z_log_var = input_args

        # Inject noise
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.,
                stddev=epsilon_std)

        return z_mean + K.exp(z_log_var / 2) * epsilon

    z = Lambda(sample, output_shape=(latent_dim,))([z_mean, z_log_var])

    decoder = Dense(intermediate_dim, activation='relu')
    reconstructer = Dense(original_dim, activation='sigmoid')

    decoded = decoder(z)
    reconstructed = reconstructer(decoded)

    vae = Model(x, reconstructed)

    reconstruction_error = original_dim * metrics.binary_crossentropy(x,
            reconstructed)

    kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var),
            axis=-1)

    vae_loss = K.mean(reconstruction_error + kl_loss)

    vae.add_loss(vae_loss)
    vae.compile(optimizer='rmsprop', loss=None)
    vae.summary()

    vae.fit(x_train, shuffle=True, epochs=epochs, batch_size=batch_size)

    # Let's visualize our latent space
    encoder = Model(x, z_mean)
    encoder.save('encoder.h5')

def test():
    encoder = load_model('encoder.h5')

    x_test_encoded = encoder.predict(x_test, batch_size=batch_size)

    color_encoding = {
            0: 'b',
            1: 'g',
            2: 'r',
            3: 'c',
            4: 'm',
            5: 'y',
            6: 'k',
            7: 'w',
            8: 'b',
            9: 'g',
    }

    y_colors = [color_encoding[y] for y in y_test]
    plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_colors)

    plt.show()

if __name__ == '__main__':
    #train()
    test()
