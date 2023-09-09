from train_model import train
from evaluate import evaluate
from predict import predict
from src.data.data_fetcher import fetch_and_save_csv

def main():
    while True:
        choice = input("Enter 'train' to train the model, 'predict' to make a prediction, or 'exit' to quit: ").lower()
        if choice == 'train':
            train()
        elif choice == 'predict':
            predict()
        elif choice == 'evaluate':
            evaluate()
        elif choice == 'fetch':
            fetch_and_save_csv()
        elif choice == 'exit':
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
