from train_model import train
from predict import predict
from evaluate_model import evaluate

def main():
    while True:
        choice = input("Enter 'train' to train the model, 'predict' to make a prediction, 'evaluate' to evaluate the model, or 'exit' to quit: ").lower()
        if choice == 'train':
            train()
        elif choice == 'predict':
            predict()
        elif choice == 'evaluate':
            evaluate()
        elif choice == 'exit':
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
