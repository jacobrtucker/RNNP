import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Concatenate, Dropout

def build_model(num_demographic_features, num_days, num_day_of_week, num_week_of_year, num_prev_proportions):
    # Inputs
    demographics_input = Input(shape=(num_demographic_features,), name="demographics")
    day_input = Input(shape=(1,), name="day_of_survey")
    day_of_week_input = Input(shape=(1,), name="day_of_week")
    week_of_year_input = Input(shape=(1,), name="week_of_year")
    prev_proportions_input = Input(shape=(num_prev_proportions,), name="previous_day_proportions")

    # Embeddings for categorical inputs
    day_embed = Embedding(input_dim=num_days, output_dim=4)(day_input)
    day_of_week_embed = Embedding(input_dim=num_day_of_week, output_dim=2)(day_of_week_input)
    week_of_year_embed = Embedding(input_dim=num_week_of_year, output_dim=2)(week_of_year_input)

    # Flatten embeddings
    day_embed_flat = tf.keras.layers.Flatten()(day_embed)
    day_of_week_flat = tf.keras.layers.Flatten()(day_of_week_embed)
    week_of_year_flat = tf.keras.layers.Flatten()(week_of_year_embed)

    # Concatenate all features
    concatenated = Concatenate()([demographics_input, day_embed_flat, day_of_week_flat, week_of_year_flat, prev_proportions_input])

    # Recurrent layer to capture temporal dependencies
    lstm_layer = LSTM(50)(tf.expand_dims(concatenated, axis=1))

    # Fully connected layers
    dense1 = Dense(100, activation='relu')(lstm_layer)
    dropout = Dropout(0.5)(dense1)
    output = Dense(num_prev_proportions, activation='softmax')(dropout)  # assuming output is a probability distribution

    # Build model
    model = Model(inputs=[demographics_input, day_input, day_of_week_input, week_of_year_input, prev_proportions_input], outputs=output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Example of dimensions, assuming some typical values
model = build_model(num_demographic_features=10, num_days=365, num_day_of_week=7, num_week_of_year=52, num_prev_proportions=5)
model.summary()
