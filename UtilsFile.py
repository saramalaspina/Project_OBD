import csv
import os

def add_csv_line(model, regularization, lambd, error, rmse, activation_fn, epochs, minibatch_size, dir):
    # Apro il file in modalità 'append' per aggiungere righe senza sovrascrivere
    with open('plots/' + dir + '/result/results.csv', mode='a', newline='') as file:
        nomi_colonne = ['model', 'regularization', 'lambda', 'error', 'rmse', 'activation_fn', 'epochs', 'minibatch_size']
        nuova_riga = {'model': model, 'regularization': regularization, 'lambda': lambd, 'error': error, 'rmse': rmse, 'activation_fn': activation_fn, 'epochs': epochs, 'minibatch_size': minibatch_size}
        writer = csv.DictWriter(file, fieldnames=nomi_colonne)

        # Scrivo l'intestazione (solo se il file è vuoto)
        if file.tell() == 0:
            writer.writeheader()

        # Aggiungo la riga passata come argomento
        writer.writerow(nuova_riga)


def clear_folder(directory):

    file_list = os.listdir(directory)

    for file_name in file_list:
        file_path = os.path.join(directory, file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)