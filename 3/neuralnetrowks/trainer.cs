using System;
using System.Collections.Generic;
using System.IO;

public class inicio
{

#region "mierda"

    private static double letterToDouble(String p)
    {
        return Convert.ToUInt16(p.ToCharArray()[0]);
    }
    
    private static double[] stringTodouble(String p)
    {
        ushort[] usA = Array.ConvertAll(p.ToCharArray(), Convert.ToUInt16);
        double[] dRet = new double[usA.Length];

        for (Int32 i = 0; i < dRet.Length; i++)
            dRet[i] = usA[i];

        return dRet;
    }

    public static myType[][] shuffle<myType>(myType[][] array)
    {
        List<myType[]> lNew = new List<myType[]>(array);
        List<myType[]> lRet = new List<myType[]>();
        Random r = new Random();
        myType[] tmp;
        Int32 iTmp;

        while (lNew.Count != 0)
        {
            iTmp = r.Next(0, lNew.Count);
            tmp = lNew[iTmp];
            lRet.Add(tmp);
            lNew.Remove(tmp);
        }

        return lRet.ToArray();
    }

    public static T[][] arrayOfArray<T>(T[] p)
    {
        T[][] aRet = new T[p.Length][];
        for (Int32 i = 0; i < p.Length; i++)
            aRet[i] = new T[1] { p[i] };
        return aRet;
    }


    public static double[] letterToArray(Int32 letterIndex)
    {
        double[] aRet = new double[27];
        aRet[letterIndex] = 1;
        return aRet;
    }

    public static double[] letterToArray(String letter)
    {
        return letterToArray(Convert.ToInt32(letterToDouble(letter)) - 97);
    }


    public static bool pseudoTrainSetUntilNumber(double[][] inputs, double[][] desired, NeuralNetwork n, double presission, Int32 samplesToLearn)
    {
        double acumSum = 0.0;
        double tmp = samplesToLearn * 2.0;
        //(1 / 2*n)*Sum( mod(desired-activation)^2) where n = number of training cases, and the sum is over every training case.
        n.pseudoTrainSet(inputs, desired, samplesToLearn);
        for (Int32 i = 0; i < samplesToLearn; i++)
        {
            acumSum = acumSum + numpy.absSqr(numpy.add(desired[i], numpy.scalar(n.Feedfordward(inputs[i]), -1.0)));

        }

        return acumSum <= (1.0/tmp)*presission;
    }


    private static void recognizeDigits(string fileName)
    {
        NeuralNetwork myNet;
        readMNist digits = new readMNist("train-labels.idx1-ubyte", "train-images.idx3-ubyte");
        Int32 samples = 100;
        Int32 samplesLearned = 0;
        
        double[][] tryiningData = new double[samples][];
        double[][] desired = new double[samples][];
        Int32 tmp = 0;
        Int32 iterations = 0;

        if(fileName.Length == 0 || (!File.Exists(fileName) && !File.Exists(fileName+"bkup")))
            myNet = new NeuralNetwork( 784, 32, 16, 10 );
        else if(File.Exists(fileName+"bkup"))
            myNet = NeuralNetwork.getFromFile(fileName+"bkup"); // new NeuralNetwork( 784, 15, 10 );
        else 
            myNet = NeuralNetwork.getFromFile(fileName);

        Console.Clear();
        bool  wannaStop=false;
        //the next logic is: if we are already trained with the set of samples, we get more, 
        while(!wannaStop){

            samplesLearned++;
            //we get one sample 
            digits.GiveNextValue(out tryiningData[samplesLearned-1], ref tmp);
            desired[samplesLearned-1]=numpy.getArrayPopulated<double>(10,0);
            desired[samplesLearned-1][tmp]=1;

            iterations = 0;
            Console.SetCursorPosition(0,samplesLearned+1);
            Console.Write("Training sample {0}...                                                                      ", samplesLearned);
            for (; !pseudoTrainSetUntilNumber(tryiningData, desired, myNet, 0.1, samplesLearned); ) // && iterations < maxIterations;)
            {
                if(Console.KeyAvailable && Console.ReadKey(true).Key == ConsoleKey.Escape){
                    myNet.saveToFile(fileName+"bkup");
                    wannaStop=true;
                    break;
                }
                myNet.FeedfordwardSet(tryiningData, desired, samplesLearned);
                iterations++;
            }
            myNet.saveToFile(fileName);

            Console.WriteLine("Learned {1} samples, took {0} iterations                                                                                 \n=================================================================", iterations, samplesLearned);

        }

        Console.SetCursorPosition(0,30);
        String sTmp;
        for (;;)
        {
            Console.WriteLine("Ingrese un número: ");
            sTmp = Console.ReadLine();
            tryiningData[0] = digits.GiveNextSpecificValue(Int32.Parse(sTmp));


            tryiningData[0] = myNet.Feedfordward(tryiningData[0]);
            
            Console.Write("Rta: ");
            for(tmp = 0; tmp < 10; tmp++)
                Console.Write(" [{0:N9}] ",tryiningData[0][tmp]);
            Console.WriteLine("");

        }

    }



    private static void recognizeAnd()
    {
        NeuralNetwork myNet = new NeuralNetwork( 2, 5, 15, 15, 1 );


        double[][] desired = new double[4][];
        double[][] dTryiningData = new double[4][];
        double[] dtmp;

        desired[0] = new double[1] { 1 };
        desired[1] = new double[1] { 1 };
        desired[2] = new double[1] { 0 };
        desired[3] = new double[1] { 0 };

        dTryiningData[0] = new double[2] { 0, 0 };
        dTryiningData[1] = new double[2] { 0, 1 };
        dTryiningData[2] = new double[2] { 1, 0 };
        dTryiningData[3] = new double[2] { 1, 1 };

        for (; myNet.Feedfordward(dTryiningData[1])[0] <= 0.99;) // || myNet.feedFordward(dTryiningData[0])[0] >= 0.7;)
        {
            myNet.pseudoTrainSet(dTryiningData, desired, dTryiningData.Length);
            myNet.FeedfordwardSet(dTryiningData, desired, dTryiningData.Length);
            //Console.WriteLine("----- ENTRENANDO: {0}", dtmp[0]);
        }

        
        string sTmp;

        for (;;)
        {
            dtmp = new double[2];
            sTmp = Console.ReadLine();
            dtmp[0] = double.Parse(sTmp);

            sTmp = Console.ReadLine();
            dtmp[1] = double.Parse(sTmp);

            dtmp = myNet.Feedfordward(dtmp);
            Console.Write("Salida personal -------{0} ", dtmp[0]);
        }

        
    }
#endregion

    public static void Main()
    {

       // readMNist a = new readMNist("train-labels.idx1-ubyte", "train-images.idx3-ubyte");
        
        //recognizeOneLetter();
        recognizeDigits("recognizeHandWritedDigits.bin");

    }
}