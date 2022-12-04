from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Flatten, Reshape, Dense, Activation, Conv2D, MaxPooling2D, Dropout, BatchNormalization, LSTM, Input, Concatenate, Embedding
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model

def deepnet1(input_dim):
    rf = Sequential()

    rf.add(Dense(512,  input_dim=input_dim))
    rf.add(BatchNormalization())
    rf.add(Activation("relu"))
    rf.add(Dropout(rate = 0.15))

    rf.add(Dense(128))
    rf.add(BatchNormalization())
    rf.add(Activation("relu"))
    rf.add(Dropout(rate = 0.15))
    rf.add(Dense(1, kernel_initializer='normal',activation='relu'))
    rf.compile(loss='mean_absolute_error', optimizer="adam", metrics=['mean_absolute_error'])

    return rf


def deepnet2(input_dim1, input_dim2):
    input_length = 4#input_length for embed

    inputs = Input(shape=(input_dim1,),name='Input_1')
    inputs2 = Input(shape=(input_length,),name='Input_2')

    embedding = Embedding(input_dim = input_dim2, output_dim = 10, input_length = input_length, name='embed')(inputs2)
    flatten = Flatten()(embedding)

    dense_emb = Dense(20)(flatten)
    bn_emb = BatchNormalization()(dense_emb)
    act_emb = Activation("relu")(bn_emb)

    concatenated = Concatenate( name='Concatenate_1')([inputs,act_emb])

    dense_all = Dense(512, name='Dense_all')(concatenated)
    bn = BatchNormalization()(dense_all)
    act = Activation("relu")(bn)
    dpo = Dropout(rate = 0.15)(act)

    dense_all2 = (Dense(128))(dpo)
    bn2 = (BatchNormalization())(dense_all2)
    act2 = (Activation("relu"))(bn2)
    dpo2 = (Dropout(rate = 0.15))(act2)

    output1 = Dense(1, kernel_initializer='normal',activation='relu')(dpo2)

    model = Model(inputs=[inputs,inputs2], outputs=output1)

    model.compile(loss='mean_absolute_error', optimizer=Adam(learning_rate = 0.001), metrics = ['mean_absolute_error'])

    return model
