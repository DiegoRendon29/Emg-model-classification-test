from TrainingModelLocally import train_model
from ModelAnalysisLocal import read_model
from ETL import *
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
def load_data_one(frame_size,subject, input_dir=None, df=None):
    # Function to create and clean the data, it can direct extract from a dataframe or create one
    if df is None:
        file = f"statistical_data_frame{frame_size}.csv"
        try:
            # Read data
            df = pd.read_csv(os.path.join(input_dir, file))
            print(df)
        except:
            print(f"File does not exist, creating a new one.")
            df = obtain_complete_data(frame_size)

    print("Loaded data")
    print("Creating dataframe")



    metrics = [
        "Census", "mean", "std", "kurtosis",
        "skewness", "entropy", "median",
        "percentile 25", "percentile 75"
    ]

    columns = [f"{metric} channel {i}" for i in range(10) for metric in metrics]

    df.fillna(0, inplace=True)

    df = df[df['subject'] == subject]
    df.drop(columns="subject", inplace=True)
    print(df.shape)
    df_validation = df[df['repetition'] == 5]
    df_train = df[(df['repetition'] != 5) & (df['repetition'] != 3)]

    df_validation.drop(columns="repetition", inplace=True)
    df_train.drop(columns="repetition", inplace=True)

    train_label = df_train.pop("movement")
    validation_label = df_validation.pop("movement")

    scaler = MinMaxScaler()
    df_train = pd.DataFrame(scaler.fit_transform(df_train), columns=df_train.columns)
    df_validation = pd.DataFrame(scaler.transform(df_validation), columns=df_validation.columns)


    train_data = df_train.values
    val_data = df_validation.values

    train_labels = tf.keras.utils.to_categorical(train_label)
    val_labels = tf.keras.utils.to_categorical(validation_label)

    idx = np.random.permutation(len(train_data))
    train_data, train_labels = train_data[idx], train_labels[idx]

    idx = np.random.permutation(len(val_data))
    val_data, val_labels = val_data[idx], val_labels[idx]
    print(f"Size of training data:{train_data.shape}")
    print(f"Size of Validation data {val_data.shape}")

    return (train_data, train_labels), (val_data, val_labels)


if __name__ == "__main__":
    frame_size = 31
    accs = []
    model_name = "ModelFrameSize31.h5"
    for i in range(1,28):
        (train_data, train_labels), (val_data, val_labels) = load_data_one(frame_size=frame_size,subject=i)
        model = read_model(model_name)
        train_model(model, train_data, train_labels, val_data, val_labels, epoch=400,batch_size=128)
        loss, accuracy = model.evaluate(val_data, val_labels)
        print(f"Subject {i}, accuracy: {accuracy}")
        accs.append(accuracy)
    print(accs)
    print(f"Promedio {sum(accs)/len(accs)}")

