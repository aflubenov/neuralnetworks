# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import numpy as np
import sys
import os
import cntk 
from cntk.device import gpu, try_set_default_device
from cntk import Trainer
from cntk.layers import Dense, Sequential, For
from cntk.learners import sgd, learning_rate_schedule, UnitType
from cntk.ops import input, sigmoid
from cntk.losses import cross_entropy_with_softmax
from cntk.metrics import classification_error
from cntk.logging import ProgressPrinter
import csv
import math
# make sure we get always the same "randomness"
np.random.seed(0)


def leerArchivoCSV(archivo, limite=-1):
    print("leyendo "+ archivo);
    baseDir = "c:\\Users\\alubenov\\Dropbox\\acciones\\rava\\"
    archivo = open(baseDir+archivo)
    archivocsv = csv.reader(archivo)
    retorno = [];
    i = 0;

    for fila in archivocsv:
        i = i + 1;
        retorno.append(fila);
        if limite != -1 and i == limite:
            return retorno;

    return retorno


def cargarData(papel, limite):
    ruedas = leerArchivoCSV("ia\\"+papel + ".csv", limite)
    operaciones = leerArchivoCSV("ia\\"+papel+"_indications.csv", -1)
    operacionesFecha = []
    operacionesDetalle = []

    ruedas.pop(0);
    operaciones.pop(0);
    operacionesFecha = np.array(operaciones)[:,0];

    #las operaciones están resumidas, indicando solo las de compra y venta,
    #tenemos que transformarlas a un arreglo tan largo como el de ruedas
    print("Generando info de compra/venta...");
    for rueda in ruedas:
        if rueda[0] in operacionesFecha:
            operacionesDetalle.append(int(operaciones[np.where(operacionesFecha==rueda[0])[0][0]][1]))
        else:
            operacionesDetalle.append(0)

    #quitamos la columna de fecha, la del volumen y la otra del daaset de ruedas
    ruedas = np.delete(np.array(ruedas), np.s_[6], axis=1) #"otra"
    ruedas = np.delete(np.array(ruedas), np.s_[5], axis=1) #volumen
    ruedas = np.delete(np.array(ruedas), np.s_[0], axis=1).astype(np.float32); #fecha
    #ahora tenemos el arreglo de ruedas sin fecha, sincronizado con el de operaciones

    return ruedas.tolist(), operacionesDetalle

##
#convierte cada rueda (apertura, maximo, 
#minimo, cierre) a una expresión de 
#porcentaje  sobre el cierre de la rueda anterior. 
# La primer rueda siempre va a tener porcentaje 0
def convertirAPorcentaje(ruedas):  #momentaneamente deprecado
    print("Convirtiendo a porcentaje...");
    retRuedas = [[0.0, 0.0, 0.0, 0.0]];
    cierreAnt = ruedas[0][3];
    l = len(ruedas);

    for i in range(1,l):
        retRuedas.append([((x / cierreAnt) - 1.0) for x in ruedas[i]]);
        cierreAnt = ruedas[i][3];
        
    return retRuedas;

##
# genera datasets con subconjuntos de las 
# últimas 'cantRuedas' ruedas, uno por 
# cada día hasta completar todas las ruedas.
def generarDatasets(ruedas, operaciones, cantRuedas):
    print("Generando datasets....");
    retDatasetRuedas = [];
    retDatasetOperaciones = [];
    l = len(ruedas);

    for i in range(cantRuedas,l+1):
        retDatasetRuedas.append(np.concatenate(ruedas[(i-cantRuedas):i], 0).tolist());
        retDatasetOperaciones.append(operaciones[i-1])

    return retDatasetRuedas, retDatasetOperaciones

##
# suma al valor de las ruedas el valor pasado por 
# parámetro, o sea, es un desplazamiento, por 
# ejemplo, si "valor" es "min(ruedas)" entonces 
# paso el grupo de ruedas a un grupo cuyo valor 
# mínimo es 0 y el máximo es la 
# diferencia entre "max(ruedas) - min(ruedas)"
def desplazarVertical(ruedas, valor):
    l = len(ruedas);
    for i in range(0, l):
        ruedas[i] = (ruedas[i] + valor);


##
# convierte el eje vertical (la cotización) a una escala logarítimica
def convertirASemilogaritmico(ruedas):
    l = len(ruedas)
    i = 0;
    minVal = min(ruedas);
    maxVal = 0.0;
    k = 0.0;
    c = 0.0;

    desplazarVertical(ruedas, (-1*minVal) + + math.exp(1.0));

    minVal = min(ruedas);
    maxVal = max(ruedas);
    rangeVal = 10.0

    # una ecuación general logarítmica es: y =  k*log(x) + c
    # queremos que los valores varíen entre 0 y "rangeVal" en forma logarítmica
    # por lo tanto, para minVal, "y" debe ser 0, y para maxVal "y" debe ser "rangeVal"

    k = rangeVal / (math.log(maxVal) - math.log(minVal));
    c = -1 * k * math.log(minVal);
   

    for i in range(0, l):
        ruedas[i] = (k * math.log(ruedas[i]) + c);

##
# Transforma un conjunto de ruedas a valores entre 0 y 1
def normalizarRangoValores(ruedas):    
    l = len(ruedas);
    i = 0;
    maxVal = max(ruedas);
    minVal = min(ruedas);
    rangeVal = float(maxVal - minVal);

    desplazarVertical(ruedas, -1*minVal);

    for i in range(0, l):
        ruedas[i] = (ruedas[i]/rangeVal);

##
# Normaliza, o sea, trata que haya igualdad de 
# situaciones repitiendo las menos frecuentes, también 
# transforma el arreglo de operaciones a un 
# vector de 3 dimensiones y mezcla los datos
def normalizarDataset(ruedas, operaciones):
    print("Normalizando.....");
    cantCompras = operaciones.count(1);
    cantVentas = operaciones.count(-1);
    cantNada = operaciones.count(0);
    l = len(operaciones);
    maximo = max(cantCompras, cantVentas,cantNada);
    cantReplicar=0;
    p=0;
    #convertimos las cantidades a factores
    cantCompras = int(maximo / cantCompras);
    cantVentas = int(maximo / cantVentas);
    cantNada = int(maximo / cantNada);

    #cada elemento para su tipo (compra, venta, nada) lo 
    #vamos a replicar tantas veces menos uno como 
    #digan las cantidades anteriores
    for i in range(0,l):
        if(operaciones[i] == 1):
            cantReplicar = cantCompras - 1;
        if(operaciones[i] == -1):
            cantReplicar = cantVentas - 1;
        if(operaciones[i] == 0):
            cantReplicar = cantNada - 1;
        for j in range(0, cantReplicar):
            operaciones.append(operaciones[i]);
            ruedas.append(ruedas[i]);

    #convertimos los 1(compra) a 1,0,0, los -1 (venta) a 0,0,1 y neutro a 0,1,0
    operaciones = [[1.0,0.0,0.0] if x == 1 else [0.0,1.0,0.0] if x == 0 else [0.0,0.0,1.0] for x in operaciones];
    
    p = np.random.permutation(len(ruedas));
    return (np.array(ruedas)[p]).tolist(), (np.array(operaciones)[p]).tolist();


# Creates and trains a feedforward classification model

def crearRed(input_dim, num_output_classes, feature):
    num_hidden_layers = 2
    hidden_layers_dim = 50

    netout = Sequential([For(range(num_hidden_layers), lambda i: Dense(hidden_layers_dim, activation=sigmoid)),
                         Dense(num_output_classes)])(feature)   
    return netout;

def cargarRedDesdeArchivo(archivo):
    input_dim = 800;
    num_output_classes = 3;

    feature = input((input_dim), np.float32);
    label = input((num_output_classes), np.float32)

    netout = crearRed(input_dim, 3, feature);
    ce = cross_entropy_with_softmax(netout, label)
    pe = classification_error(netout, label)

    lr_per_minibatch=learning_rate_schedule(0.5, UnitType.minibatch)
    # Instantiate the trainer object to drive the model training
    learner = sgd(netout.parameters, lr=lr_per_minibatch)
    progress_printer = ProgressPrinter(1)
    trainer = Trainer(netout, (ce, pe), learner, progress_printer)


    trainer.restore_from_checkpoint(archivo);

    return netout;


def entrenar(checkpoint, entrRuedas, entrOperaciones, input_dim, num_output_classes, testRuedas, testOperaciones):
    minibatch_size = 100;
    epocs=900;
    minibatchIteraciones = int(len(entrOperaciones) / minibatch_size);

    # Input variables denoting the features and label data
    feature = input((input_dim), np.float32)
    label = input((num_output_classes), np.float32)

    netout = crearRed(input_dim, num_output_classes, feature);

    ce = cross_entropy_with_softmax(netout, label)
    pe = classification_error(netout, label)

    lr_per_minibatch=learning_rate_schedule(0.25, UnitType.minibatch)
    # Instantiate the trainer object to drive the model training
    learner = sgd(netout.parameters, lr=lr_per_minibatch)
    progress_printer = ProgressPrinter(log_to_file=checkpoint+".log", num_epochs=epocs);
    trainer = Trainer(netout, (ce, pe), learner, progress_printer)


    if os.path.isfile(checkpoint):
        trainer.restore_from_checkpoint(checkpoint);

    npentrRuedas = np.array(entrRuedas).astype(np.float32);
    npentrOperaciones = np.array(entrOperaciones).astype(np.float32);

    #iteramos una vez por cada "epoc"
    for i in range(0, epocs):
        p = np.random.permutation(len(entrRuedas));
        npentrOperaciones = npentrOperaciones[p];
        npentrRuedas = npentrRuedas[p];

        #ahora partimos los datos en "minibatches" y entrenamos
        for j in range(0, minibatchIteraciones):
            features = npentrRuedas[j*minibatch_size:(j+1)*minibatch_size];
            labels = npentrOperaciones[j*minibatch_size:(j+1)*minibatch_size];
            trainer.train_minibatch({feature: features, label: labels});
        trainer.summarize_training_progress()
        
    
    trainer.save_checkpoint(checkpoint);



    minibatchIteraciones = int(len(testOperaciones) / minibatch_size);
    avg_error = 0;
    for j in range(0, minibatchIteraciones):

        test_features = np.array(testRuedas[j*minibatch_size:(j+1)*minibatch_size]).astype(np.float32);
        test_labels = np.array(testOperaciones[j*minibatch_size:(j+1)*minibatch_size]).astype(np.float32);
        #test_features = np.array( entrRuedas[0:minibatch_size]).astype(np.float32);
        #test_labels = np.array(entrOperaciones[0:minibatch_size]).astype(np.float32);
        avg_error = avg_error + ( trainer.test_minibatch(
            {feature: test_features, label: test_labels}) / minibatchIteraciones)

    return avg_error

def correrEntrenador(papel):
    lared="";
    ruedas, operaciones = cargarData(papel, -1);
    #ruedas = convertirAPorcentaje(ruedas);
    ruedas, operaciones = generarDatasets(ruedas, operaciones,200);

    print("Poniendo todo entre 0 y 1 ...");
    for i in range(0, len(ruedas)):
        convertirASemilogaritmico(ruedas[i]);
        normalizarRangoValores(ruedas[i]);

    ruedas, operaciones = normalizarDataset(ruedas, operaciones);
    
    #dividimos 70% - 30% los datos para que sean entrenamiento y test respectivamente
    setenta = int(len(ruedas) * 0.7);

    test_ruedas = ruedas[setenta:len(ruedas)];
    test_operaciones = operaciones[setenta:len(ruedas)]

    ruedas = ruedas[0:setenta];
    operaciones = operaciones[0:setenta];

  #  laredEntrenador = crearRed(800,3);
    
    error = entrenar(papel + ".dnn", ruedas, operaciones, 800, 3, test_ruedas, test_operaciones);
    

    print(" error rate on an unseen minibatch %f" % error)


def evaluador(papel, cantRuedas):
    netout = cargarRedDesdeArchivo(papel + ".dnn");
    ruedas = leerArchivoCSV(papel + ".csv");
    
    ruedas.pop(0);
    ruedas = np.array(ruedas);
    fechas = ruedas[:,0];
    fechas = fechas.tolist();
    
    ruedas = np.delete(ruedas, np.s_[6], axis=1) #"otra"
    ruedas = np.delete(ruedas, np.s_[5], axis=1) #volumen
    ruedas = np.delete(ruedas, np.s_[0], axis=1).astype(np.float32); #fecha
 

    #ruedas = convertirAPorcentaje(ruedas.tolist());
    print("Poniendo todo entre 0 y 1 ...");
    for i in range(0, len(ruedas)):
        convertirASemilogaritmico(ruedas[i]);
        normalizarRangoValores(ruedas[i]);

    #vamos a traer las últimas 30 ruedas y vamos a mostrar lo que hacemos en cada día
    indiceFecha = len(fechas) -1;
    verbal = ["COMPRAR","nada","VENDER"];

    print("FECHA (AAAA-MM-DD)    OPERACION");
    print("-------------------------------");

    for i in range(0,cantRuedas):
        ruedita = np.concatenate(ruedas[indiceFecha-200+1:indiceFecha+1], 0).astype(np.float32);
        quehacer = netout.eval(ruedita).tolist()[0];

        maximo = max(quehacer);
        print( fechas[indiceFecha] + "        "+ verbal[quehacer.index(maximo)]);
        indiceFecha = indiceFecha - 1;
   




if __name__ == '__main__':
    # Specify the target device to be used for computing, if you do not want to
    # use the best available one, e.g.
    # try_set_default_device(cpu())


    #error = ffnet()
    correrEntrenador("bma")

##    evaluador(sys.argv[1]);

