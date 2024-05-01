import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Concatenate, Dropout
import tensorflow_probability as tfp
import os
import datetime
dir_path = os.path.dirname(os.path.realpath(__file__))

class RNNP:
    model = None
    num_outcomes = None
    outcome_dims = None
    posp = {}

    def __init__(self, demo_feature_names, num_demographic_features, num_days, num_outcomes, outcome_dims):
        # Inputs
        # demographics_input = Input(shape=(num_demographic_features,), name="demographics")
        day_input = Input(shape=(1,), name="day_of_survey")
        day_of_week_input = Input(shape=(1,), name="day_of_week")
        week_of_year_input = Input(shape=(1,), name="week_of_year")

        inputs = [day_input, day_of_week_input, week_of_year_input]

        # Embeddings for categorical inputs
        day_embed = Embedding(input_dim=num_days, output_dim=4)(day_input)
        day_of_week_embed = Embedding(input_dim=7, output_dim=2)(day_of_week_input)
        week_of_year_embed = Embedding(input_dim=52, output_dim=2)(week_of_year_input)

        # Flatten embeddings
        day_embed_flat = tf.keras.layers.Flatten()(day_embed)
        day_of_week_flat = tf.keras.layers.Flatten()(day_of_week_embed)
        week_of_year_flat = tf.keras.layers.Flatten()(week_of_year_embed)

        features = [day_embed_flat, day_of_week_flat, week_of_year_flat]
        for dem in demo_feature_names:
            num_dims = round(num_demographic_features[dem] ** 0.25)
            if num_dims < 1:
                num_dims = 1
            inputs.append(Input(shape=(1,), name=dem))
            features.append(tf.keras.layers.Flatten()(Embedding(input_dim=num_demographic_features[dem],output_dim=num_dims)(inputs[-1])))

        # Concatenate all features
        concatenated = Concatenate()(features)

        # Recurrent layer to capture temporal dependencies
        lstm_layer = LSTM(100)(tf.expand_dims(concatenated, axis=1))

        # Fully connected layers
        dense1 = Dense(100, activation='relu')(lstm_layer)
        dropout = Dropout(0.5)(dense1)
        dense2 = Dense(100, activation='relu')(dropout)
        dropout2 = Dropout(0.5)(dense2)
        output = Dense(sum(outcome_dims), activation='sigmoid')(dropout2)  # assuming output is a probability distribution

        # Build model
        model = Model(inputs=inputs, outputs=output)
        model.compile(optimizer='adam', loss=self.negDirLL, metrics=['accuracy'])
        
        self.model = model
        self.num_outcomes = num_outcomes
        self.outcome_dims = outcome_dims

        for od in range(num_outcomes):
            self.posp[od] = []
            for i in range(10000):
                self.posp[od].append([tfp.distributions.Uniform(low=0.0, high=1.0) for j in range(outcome_dims[od])])
    
    def fit(self, X_train, y_train, epochs=10, batch_size=32):
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
    
    def negDirLL(self, y_true, y_pred):
        total_counts = y_true[:-self.num_outcomes]
        nll = float(0)
        start_index = 0
        for i in range(self.num_outcomes):
            end_index = start_index + self.outcome_dims[i]
            y = y_true[start_index:end_index]
            yp = y_pred[start_index:end_index]
            yp = tf.softmax(yp)
            for posp in self.posp[i]:
                nll += -tfp.distributions.Dirichlet(posp).log_prob(yp) - tfp.distributions.Multinomial(total_counts[i], probs=yp).log_prob(y)
            start_index = end_index
        return nll

    def save(self, path=None):
        if path is None:
            path = os.path.join(dir_path, 'modelcache/RNNP-' + str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")))
        self.model.save(path)