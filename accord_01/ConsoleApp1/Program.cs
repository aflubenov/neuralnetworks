using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Accord.Controls;
using Accord.IO;
using Accord.Math;
using Accord.Statistics.Distributions.Univariate;
using Accord.MachineLearning.Bayes;
using Accord.Neuro;
using System.Data;
using System.IO;
using Accord.Neuro.Learning;

namespace ConsoleApp1
{
    class Program
    {


        /// <summary>
        /// 
        /// </summary>
        /// <param name="datasetRuedas"></param>
        /// <param name="datasetResultados"></param>
        /// <param name="etiquetasFechas"></param>
        /// <param name="cantRuedas"></param>
        /// <param name="archivoCsv"></param>
        /// <param name="Samples">-1 if we want all the data</param>
        static void cargarDatosYResultados(out double[][] datasetRuedas, out double[] datasetResultados, out string[] etiquetasFechas, Int32 cantRuedas, string archivoCsv, Int32 Samples)
        {
            CsvReader tmpReader;


            String[][] ruedasFechas, tmpIndicacionesCompraFechas;
            Int32[][] tmpIndicacionesCompraCompra;
            Int32 i, j;
            double[][] data;
            double dataAnt;
            double[] resultadosCompra;



            //obtenemos las fechas
            tmpReader = new CsvReader("c:\\Users\\alubenov\\Dropbox\\acciones\\rava\\" + archivoCsv + ".csv", true);
            ruedasFechas = tmpReader.ToTable(new[] { "fecha" }).ToArray<String>();
            tmpReader.Close();

            //obtenemos la data de la rueda (valor inicial, máximo, mínimo, valor de cierre y volumen operado)
            tmpReader = new CsvReader("c:\\Users\\alubenov\\Dropbox\\acciones\\rava\\" + archivoCsv + ".csv", true);
            data = tmpReader.ToTable(new[] { "apertura", "maximo", "minimo", "cierre" }).ToArray<double>();
            tmpReader.Close();


            //obtenemos las indicaciones: fecha y si compro o no
            tmpReader = new CsvReader("c:\\Users\\alubenov\\Dropbox\\acciones\\rava\\ia\\" + archivoCsv + "_indications.csv", true);
            tmpIndicacionesCompraFechas = tmpReader.ToTable(new[] { "fecha"}).ToArray<string>();
            tmpReader.Close();
            tmpReader = new CsvReader("c:\\Users\\alubenov\\Dropbox\\acciones\\rava\\ia\\" + archivoCsv + "_indications.csv", true);
            tmpIndicacionesCompraCompra = tmpReader.ToTable(new[] { "compra" }).ToArray<Int32>();

            //le ponemos la indicación a cada rueda: 0 si no hay nada que hacer, 1 si compramos, -1 si vendemos
            resultadosCompra = new double[ruedasFechas.Length];
            
            for(i = 0; i < resultadosCompra.Length; i++)
            {
                for (j = 0; j < tmpIndicacionesCompraCompra.Length; j++)
                    if (tmpIndicacionesCompraFechas[j][0] == ruedasFechas[i][0])
                        break;

                if (j == tmpIndicacionesCompraCompra.Length)
                    resultadosCompra[i] = 0.0;
                else
                    resultadosCompra[i] = tmpIndicacionesCompraCompra[j][0];

            }

            //pasamos todos los valores de las ruedas a porcentajes respecto del valor de cierre anterior
            dataAnt = data[0][3];
            for (i = 0; i < data[0].Length; i++) data[0][i] = 0.0;

            for(i = 1; i < data.Length; i++)
            {
                double tmp = data[i][3];
                for (j = 0; j < data[i].Length; j++) data[i][j] = ((data[i][j] / dataAnt)-1.0);
                dataAnt = tmp;
            }
            ;


            etiquetasFechas = new string[ruedasFechas.Length - cantRuedas];
            Array.Copy(ruedasFechas.Apply((p) => p[0]), cantRuedas, etiquetasFechas, 0, etiquetasFechas.Length);
            if(Samples != -1)
                generarDatasets(data.Take(Samples).ToArray(), resultadosCompra.Take(Samples).ToArray(), out datasetRuedas, out datasetResultados, cantRuedas);
            else
                generarDatasets(data, resultadosCompra, out datasetRuedas, out datasetResultados, cantRuedas);

            normalizeDataset(datasetRuedas, datasetResultados, out datasetRuedas, out datasetResultados);

            

        }

        static void normalizeDataset(double[][] ruedas, double[] resultados, out double[][] datasetRuedas, out double [] datasetResultados)
        {
            List<double[]> lRuedas = new List<double[]>(ruedas);
            List<double> lResultados = new List<double>(resultados);
            double[] lTmp;
            double[] lValoresDistintos;
            double masFrecuente = -1;
            Int32 cantidad = 0, iTmp, i, cantRepetir;

            //obtenemos los datos distintos en los resultados
            lValoresDistintos = lResultados.Distinct().ToArray();

            //ahora los contamos y obtenemos el más frecuente
            for(i = 0; i < lValoresDistintos.Length; i++)
            {
                iTmp = lResultados.FindAll((p) => p == lValoresDistintos[i]).Count();
                if(iTmp > cantidad)
                {
                    cantidad = iTmp;
                    masFrecuente = lValoresDistintos[i];
                }
            }

            //en base al más frecuente, repetimos datos de los otros
            iTmp = lResultados.Count();
            for (i = 0; i < lValoresDistintos.Length; i++)
            {
                if (lValoresDistintos[i] == masFrecuente)
                    continue;

                cantRepetir = cantidad / lResultados.FindAll((p) => p == lValoresDistintos[i]).Count();

                if (cantRepetir < 2)
                    continue;

                cantRepetir = cantRepetir - 1;
                for(Int32 j = iTmp - 1; j >= 0; j--)
                {
                    if (lResultados[j] != lValoresDistintos[i])
                        continue;

                    for(Int32 k = 0; k < cantRepetir; k++)
                    {
                        lResultados.Add(lResultados[j]);
                        lRuedas.Add(lRuedas[j]);
                    }
                }
                
            }

            datasetRuedas = lRuedas.ToArray();
            datasetResultados = lResultados.ToArray();
            
        }

        static void generarDatasets(double[][] ruedas, double[] resultados, out double[][] datasetRuedas, out double[] datasetResultados, Int32 cantRuedas)
        {
            //vamos a hacer conjuntos que contendrán las últimas 'cantidadRuedas" ruedas, uno por cada día partiendo del 'cantidadRuedas'-esimo día
            Int32 cantDatasets = ruedas.Length - cantRuedas;
            Int32 k;

            datasetRuedas = new double[cantDatasets][];
            datasetResultados = new double[cantDatasets];

            for (Int32 i = 0; i < cantDatasets; i++)
            {
                datasetRuedas[i] = new double[cantRuedas * 4];
                datasetResultados[i] = resultados[i + cantRuedas - 1];
                k = 0;
                for (Int32 j = 0; j < cantRuedas; j++)
                {
                    datasetRuedas[i][k] = ruedas[i + j][0]; k++;
                    datasetRuedas[i][k] = ruedas[i + j][1]; k++;
                    datasetRuedas[i][k] = ruedas[i + j][2]; k++;
                    datasetRuedas[i][k] = ruedas[i + j][3]; k++;
                    //datasetRuedas[i][k] = ruedas[i + j][4]; k++;
                }
                
            }

        }


        public static myType[][] shuffle<myType>(myType[][] arrayData, ref myType[][] outArrayResult)
        {
            List<myType[]> lNew = new List<myType[]>(arrayData);
            List<myType[]> lRet = new List<myType[]>();

            List<myType[]> lNewResult = new List<myType[]>(outArrayResult);
            List<myType[]> lRetResult = new List<myType[]>();

            Random r = new Random(System.DateTime.Now.Millisecond);
            myType[] tmp;
            Int32 iTmp;

            while (lNew.Count != 0)
            {
                iTmp = r.Next(0, lNew.Count);
                tmp = lNew[iTmp];
                lRet.Add(tmp);
                lNew.Remove(tmp);

                tmp = lNewResult[iTmp];
                lRetResult.Add(tmp);
                lNewResult.Remove(tmp);

                outArrayResult[lRetResult.Count - 1] = lRetResult[lRetResult.Count - 1];
            }

            return lRet.ToArray();
        }


        public static myType[][] shuffle<myType>(myType[][] arrayData, ref myType[] outArrayResult)
        {
            List<myType[]> lNew = new List<myType[]>(arrayData);
            List<myType[]> lRet = new List<myType[]>();

            List<myType> lNewResult = new List<myType>(outArrayResult);
            List<myType> lRetResult = new List<myType>();

            Random r = new Random(System.DateTime.Now.Millisecond);
            myType[] tmp;
            myType tmp2;
            Int32 iTmp;

            while (lNew.Count != 0)
            {
                iTmp = r.Next(0, lNew.Count);
                tmp = lNew[iTmp];
                lRet.Add(tmp);
                lNew.Remove(tmp);

                tmp2 = lNewResult[iTmp];
                lRetResult.Add(tmp2);
                lNewResult.Remove(tmp2);

                outArrayResult[lRetResult.Count - 1] = lRetResult[lRetResult.Count - 1];
            }

            return lRet.ToArray();
        }


        static void writeLog(String data, String filename)
        {
            try
            {
                using (StreamWriter sw = File.AppendText(filename))
                {
                    sw.WriteLine(data);
                }
            }
            catch { }
        }


        static double Train (ISupervisedLearning teacher, Int32 minibatches, Int32 miniBatchesIteraciones, Int32 epocs, double iterarHastaError, double[][] data, double[][] resultados)
        {
            Int32 tamMinibatch = data.Length / minibatches;
            Int32 iteracion = 0;
            double[][] minibatch = new double[tamMinibatch][];
            double[][] minibatchResultados = new double[tamMinibatch][];
            double totalError = 0;

            do
            {
                // Compute one learning iteration
                iteracion++;
              //  data = shuffle<double>(data, ref resultados);

                totalError = 0.0;
                for (Int32 i = 0; i < minibatches; i++)
                {
                    Array.Copy(data, i * tamMinibatch, minibatch, 0, tamMinibatch);
                    Array.Copy(resultados, i * tamMinibatch, minibatchResultados, 0, tamMinibatch);

                    //corremos la cantidad indicada menos uno, así la otra la corremos y obtenemos el error
                    for (Int32 j = 0; j < miniBatchesIteraciones - 1; j++)
                        teacher.RunEpoch(minibatch, minibatchResultados);
                    //corremos la iteración faltante para obtener el error.
                    totalError += (teacher.RunEpoch(minibatch, minibatchResultados) / tamMinibatch);
                }

                //dividimos el error por la cantidad de minibatches y lo reportamos
                totalError = totalError / (double)minibatches;
                Console.WriteLine(String.Format("------------EPOC {0} ERROR: {1} -------------", iteracion, totalError), "log.csv");
                writeLog(String.Format("{0}, {1}", iteracion, totalError), "log.csv");

            } while (iteracion < epocs && totalError > iterarHastaError); //epoch

            return totalError;
        }

        static void Main(string[] args)
        {

            double[][] datasetRuedas;
            double[] datasetResultadosMonodimensional;
            double[][] datasetResultados;
            string[] etiquetasFechas;
            Int32 cantRuedas = 200;
            Int32 entrenamientoCant, testCant;

            double[][] entrenamientoRuedas, testRuedas;
            double[][] entrenamientoResults, testResults;

            cargarDatosYResultados(out datasetRuedas, out datasetResultadosMonodimensional, out etiquetasFechas,  cantRuedas, "ggal",250);

            datasetRuedas = shuffle<double>(datasetRuedas, ref datasetResultadosMonodimensional);

            datasetResultados = new double[datasetResultadosMonodimensional.Length][];
            for(var i = 0; i < datasetResultadosMonodimensional.Length; i++)
            {
                switch (datasetResultadosMonodimensional[i])
                {
                    case 0:
                        datasetResultados[i] = new double[] { 0, 1, 0 };
                        break;
                    case 1:
                        datasetResultados[i] = new double[] { 1, 0, 0 };
                        break;
                    case -1:
                        datasetResultados[i] = new double[] { 0, 0, 1 };
                        break;
                }
            }


            entrenamientoCant = (Int32) (datasetRuedas.Length * 0.7);
            testCant = datasetRuedas.Length - entrenamientoCant;

            entrenamientoRuedas = new double[entrenamientoCant][];
            entrenamientoResults = new double[entrenamientoCant][];
            testRuedas = new double[testCant][];
            testResults = new double[testCant][];

            Array.Copy(datasetRuedas, entrenamientoRuedas, entrenamientoCant);
            Array.Copy(datasetResultados, entrenamientoResults, entrenamientoCant);
            Array.Copy(datasetRuedas, entrenamientoCant, testRuedas, 0, testCant);
            Array.Copy(datasetResultados, entrenamientoCant, testResults, 0, testCant);

   
            IActivationFunction activationFunction =  new SigmoidFunction(); //new BipolarSigmoidFunction(); //

            ActivationNetwork network;


            if (File.Exists("nn.bin"))
                network = (ActivationNetwork)ActivationNetwork.Load("nn.bin");
            else
            {
                network = new ActivationNetwork(activationFunction, inputsCount: entrenamientoRuedas[0].Length, neuronsCount: new[] {30, 3 //la salida
                                                                                                                                    });
                network.Randomize();
            }

            // Create a Levenberg-Marquardt algorithm
            var teacher = new Accord.Neuro.Learning.ParallelResilientBackpropagationLearning(network);
            /*{
                DecreaseFactor= 0.1,
                IncreaseFactor = 5
                
            }

            var teacher = new Accord.Neuro.Learning.BackPropagationLearning(network)
            {
               
               LearningRate = 0.01,
               Momentum = 0
            }*/
            ;

      
            File.Delete("log.csv");
            writeLog("Iteracion, error", "log.csv");

            
            Train(teacher, entrenamientoRuedas.Length / 5, 10, 5000, 0.05, entrenamientoRuedas, entrenamientoResults);
       

            network.Save("nn.bin");
            
            
            // Plot the results

            //DataBarBox.Show(etiquetasFechas, entrenamientoResults);
            Random rndNumero = new Random(System.DateTime.Now.Millisecond);
            Int32[] indices = new Int32[3];
            
            for(Int32 i = 0; i < 3; i++)
            {
                indices[i] = rndNumero.Next(0, testRuedas.Length);
            }

            double[][] answers = indices.Apply((p) => network.Compute(testRuedas[p])); // entrenamientoRuedas.Apply(network.Compute);
            double[][] testdata = indices.Apply((p) => testResults[p]);

            for (Int32 i = 0; i < answers.Length; i++)
            {
                Console.WriteLine(String.Format(" {0}: Test: {1} , {2} , {3}   --- Respuesta: {4}  / {5}  / {6}", i, testdata[i][0], testdata[i][1], testdata[i][2], answers[i][0], answers[i][1], answers[i][2]));
                Int32 imax = 0;
                //convertimos el mayor valor de la respuesta a 1 y el resto a 0
                for(Int32 j = 0; j < answers[i].Length; j++)
                    if (answers[i][j] > answers[i][imax]) imax = j;

                for (Int32 j = 0; j < answers[i].Length; j++)
                    answers[i][j] = (j == imax ? 1 : 0);
            }
            
            
            DataBarBox.Show(new String[] { "1", "2", "3" }, testdata);
            DataBarBox.Show(new String[] { "1", "2", "3" }, answers).Hold();
            
            //.Hold();


        }
    }
}
