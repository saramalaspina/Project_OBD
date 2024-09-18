import csv
import os

def add_csv_line(model, regularization, lambd, error, rmse, activation_fn,  dir):
    # Apro il file in modalità 'append' per aggiungere righe senza sovrascrivere
    with open('plots/' + dir + '/' + activation_fn + '/results.csv', mode='a', newline='') as file:
        nomi_colonne = ['model', 'regularization', 'lambda', 'rmse', 'activation_fn']
        nuova_riga = {'model': model, 'regularization': regularization, 'lambda': lambd, 'error': error, 'rmse': rmse, 'activation_fn': activation_fn}
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