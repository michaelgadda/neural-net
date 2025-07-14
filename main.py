from src.network import Network
from sklearn.datasets import load_digits 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


def main():
    digits = load_digits()
    X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, random_state = 20)
    scaler = StandardScaler()          # create the transformer
    X_train_std = scaler.fit_transform(X_train)   # fit on TRAIN, then transform
    X_test_std  = scaler.transform(X_test)  
    NN = Network(learning_rate=.01)
    print(X_train_std.shape)
    NN.add_input_layer(X_train_std)
    NN.add_hidden_layer(X_train_std.shape[1], 13, "relu")
    NN.add_hidden_layer(13, 12, "relu")
    NN.add_hidden_layer(12, 11, "relu")
    NN.add_hidden_layer(11, 10, "relu")
    NN.add_output_layer(10, 10, y_train, True, "softmax")
    NN.add_loss("negative_log_loss", y_train)
    for i in range(5000):
        NN.forward()
        NN.backward()
    NN.forward(X_test_std, y_test)


    print("Accuracy:", accuracy_score(y_test, NN.predictions))
if __name__ == '__main__':
    main()