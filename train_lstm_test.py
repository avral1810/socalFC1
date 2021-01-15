import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as klearn
import os
import matplotlib.pyplot as plt

steps_of_history = 10


def get_model_movement():
    # Network building
    model = tf.keras.Sequential([
        klearn.InputLayer(input_shape=[10, 128], name='net1_layer1'),
        klearn.LSTM(units=256, return_sequences=True, name='net1_layer2'),
        klearn.Dropout(0.6, name='net1_layer3'),
        klearn.LSTM(units=256, return_sequences=False, name='net1_layer4'),
        klearn.Dropout(0.6, name='net1_layer5'),
        klearn.Flatten(),
        klearn.Dense(5, activation='softmax', name='net1_layer6')
    ])
    opt = tf.keras.optimizers.SGD(learning_rate=0.001,clipvalue=5.0)
    model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])
    return model


def get_model_action():
    # Network building
    model = tf.keras.Sequential([
        klearn.InputLayer(input_shape=[10, 128], name='net2_layer1'),
        klearn.LSTM(units=256, return_sequences=True, name='net2_layer2'),
        klearn.Dropout(0.6, name='net2_layer3'),
        klearn.LSTM(units=256, return_sequences=False, name='net2_layer4'),
        klearn.Dropout(0.6, name='net2_layer5'),
        klearn.Flatten(),
        klearn.Dense(5, activation='softmax', name='net1_layer6')
    ])
    opt = tf.keras.optimizers.SGD(learning_rate=0.001, clipvalue=5.0)
    model.compile(optimizer=opt, loss='categorical_crossentropy',metrics=['accuracy'])
    return model


def reshape_for_lstm(data):
    trainX = []
    trainY_movement = []
    trainY_action = []

    for i in range(0, len(data) - steps_of_history):
        window = data[i:i + steps_of_history]

        sampleX = []
        for row in window:
            sampleX.append(row[0])
        sampleY_movement = np.array(window[-1][1]).reshape(-1)
        sampleY_action = np.array(window[-1][2]).reshape(-1)

        trainX.append(np.array(sampleX).reshape(steps_of_history, -1))
        trainY_movement.append(sampleY_movement)
        trainY_action.append(sampleY_action)

    print(np.array(trainX).shape)
    print(np.array(trainY_movement).shape)
    print(np.array(trainY_action).shape)

    return np.array(trainX), np.array(trainY_movement), np.array(trainY_action)


def get_list():
    list = []
    n_samples = 10000
    for i in range(0, n_samples):
        feature_vector = np.random.rand(128, 1)
        output_movement = np.zeros((5, 1))
        output_movement[np.random.randint(0, 4), 0] = 1
        output_action = np.zeros((5, 1))
        output_action[np.random.randint(0, 4), 0] = 1
        list.append([feature_vector, output_movement, output_action])
    return list



def main_all():
    training_all = np.zeros(shape=(0, 3))
    for filename in os.listdir('rnn'):
        filename = 'rnn/' + filename
        d = np.load(filename,allow_pickle=True)
        training_all = np.concatenate((training_all, d))

    data = list(training_all)
    trainX, trainY_movement, trainY_action = reshape_for_lstm(data)
    print(trainX.shape)
    with tf.Graph().as_default():
        # model_movement = tf.keras.models.load_model('./fifa_models2/model_movement')
        # opt = tf.keras.optimizers.SGD(learning_rate=0.001, clipvalue=5.0)
        # model_movement.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        model_movement = get_model_movement()
        history = model_movement.fit(trainX, trainY_movement, epochs=150, validation_split=0.35)
        # list all data in history
        print(history.history.keys())
        # summarize history for accuracy
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Movement Accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        plt.savefig('fifa_models2/model_movement/MovementAccuracy.png')
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Movement Loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        plt.savefig('fifa_models2/model_movement/MovementLoss.png')
        model_movement.save('fifa_models2/model_movement')

    # with tf.Graph().as_default():
    #     # model_action = tf.keras.models.load_model('./fifa_models2/model_action')
    #     # opt = tf.keras.optimizers.SGD(learning_rate=0.001, clipvalue=5.0)
    #     # model_action.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    #     model_action = get_model_action()
    #     history = model_action.fit(trainX, trainY_action, epochs=150, validation_split=0.35)
    #     plt.plot(history.history['accuracy'])
    #     plt.plot(history.history['val_accuracy'])
    #     plt.title('Action Accuracy')
    #     plt.ylabel('accuracy')
    #     plt.xlabel('epoch')
    #     plt.legend(['train', 'test'], loc='upper left')
    #     plt.show()
    #     plt.savefig('fifa_models2/model_action/ActionAccuracy.png')
    #     # summarize history for loss
    #     plt.plot(history.history['loss'])
    #     plt.plot(history.history['val_loss'])
    #     plt.title('Action Loss')
    #     plt.ylabel('loss')
    #     plt.xlabel('epoch')
    #     plt.legend(['train', 'test'], loc='upper left')
    #     plt.show()
    #     plt.savefig('fifa_models2/model_action/ActionLoss.png')
    #     model_action.save('fifa_models2/model_action')

    return


main_all()
