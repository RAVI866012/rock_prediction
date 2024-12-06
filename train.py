from src.data_preprocessing import load_images, split_data
from src.model import create_model
from tensorflow.keras.callbacks import ModelCheckpoint

def train_model():
    images, labels = load_images('data/train')
    X_train, X_test, y_train, y_test = split_data(images, labels)
    
    model = create_model()
    checkpoint = ModelCheckpoint('models/rock_model.keras', save_best_only=True, monitor='val_loss')
    
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32, callbacks=[checkpoint])
    
    model.save('models/rock_model.keras')

if __name__ == '__main__':
    train_model()
