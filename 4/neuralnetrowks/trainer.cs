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
        n.pseudoTrainSet(inputs, desired);
        for (Int32 i = 0; i < samplesToLearn; i++)
        {
            acumSum = acumSum + numpy.absSqr(numpy.add(desired[i], numpy.scalar(n.Feedfordward(inputs[i]), -1.0)));

        }

        return acumSum <= (1.0/tmp)*presission;
    }

    public static bool pseudoTrainSetByEpocs(double[][] inputs, double[][] desired, NeuralNetwork n, double presission, Int32 epocs, string fileName)
    {
        Int32[][] indexes = new Int32[inputs.Length][];
        double[][] aEpocsInputs = new double[epocs][];
        double[][] aEpocsDesired = new double[epocs][];
        Int32 i = 0, l = inputs.Length;
        Int32 nEpocs = l/epocs;
        Int32 iterations = nEpocs*150;
        Int32 epocIterations = 0;
        Int32 lastTotalIterations = 0;

        if(l % epocs != 0){
            Console.WriteLine("ahhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh not multiples!!!!");
            return false;
        }


        //we set this to suffle indexes
        for(i = 0; i < l; i ++)
            indexes[i] = new Int32[1]{i};

        //now we are to train by epocs
        while(iterations > nEpocs*100){
            lastTotalIterations = iterations;
            iterations = 0;
            //suffling inputs
            indexes = shuffle<Int32>(indexes);

            for(i = 0; i < nEpocs; i ++){
                //creating the epoc
                for(Int32 j = 0; j < epocs; j ++){
                    aEpocsInputs[j] = inputs[indexes[i*epocs+j][0]];
                    aEpocsDesired[j] = desired[indexes[i*epocs+j][0]];
                }

                //we train it
                epocIterations = 0;
                for(;!pseudoTrainSetUntilNumber(aEpocsInputs, aEpocsDesired, n, presission, epocs);){
                    if(Console.KeyAvailable && Console.ReadKey(true).Key == ConsoleKey.Escape)
                        return false;

                    n.FeedfordwardSet(aEpocsInputs, aEpocsDesired, epocs);
                    Console.SetCursorPosition(0,epocs+3);
                    iterations ++;
                    epocIterations++;
                    Console.SetCursorPosition(0,epocs+3);
                    Console.WriteLine("Training Epoc # {0} of {1}, {3} Iteration ({2} total iterations so far)....",i,nEpocs, iterations, epocIterations );
/*
                    if(i == 0 && iterations > lastTotalIterations/2){
                        Console.WriteLine("Toooo much first iteration, skiping...." );                        
                        break;
                    } else*/
                     if(epocIterations > 150){ // 200 && epocIterations > (iterations / (i+1)) ){
                                Console.WriteLine("Skiping Epoc #{0} due to a lot of iterations ({1}))....",i,epocIterations );                        
                                break;
                    }

                }
                n.saveToFile(fileName+"bkup");    
            }
            n.saveToFile(fileName);
        }

        return true;
        
    }


    private static void recognizeDigits(string fileName, Int32 samples, Int32 epocs, double presission)
    {
        NeuralNetwork myNet;
        readMNist digits = new readMNist("train-labels.idx1-ubyte", "train-images.idx3-ubyte");
        //Int32 samples = 1000;
        //Int32 epocs = 5;
        Int32 samplesLearned = 0;
        
        double[][] tryiningData = new double[samples][];
        double[][] desired = new double[samples][];
        List<double[]> lTryingdata = new List<double[]>();
        List<double[]> lDesired = new List<double[]>();
        Int32 tmp = 0;
//        Int32 iterations = 0;

        if(fileName.Length == 0 || (!File.Exists(fileName) && !File.Exists(fileName+"bkup")))
            myNet = new NeuralNetwork( 784, 40, 20, 10 );
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
            for(Int32 i = 0; i < samples; i ++){
                digits.GiveNextValue(out tryiningData[i], ref tmp);
                desired[i]=numpy.getArrayPopulated<double>(10,0);
                desired[i][tmp]=1;
            }

            lTryingdata.AddRange(tryiningData);
            lDesired.AddRange(desired);

            Console.SetCursorPosition(0,epocs+10);
            Console.WriteLine("Training sample set #{0}...", samplesLearned );
            
            wannaStop = !pseudoTrainSetByEpocs(lTryingdata.ToArray(), lDesired.ToArray(), myNet, presission, epocs, fileName);
                

/*            iterations = 0;
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
*/            
            myNet.saveToFile(fileName+"bkup");
            myNet.saveToFile(fileName);

//            Console.WriteLine("Learned {1} samples, took {0} iterations                                                                                 \n=================================================================", iterations, samplesLearned);

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



  
#endregion

    public static void Main(string[] args)
    {

       // readMNist a = new readMNist("train-labels.idx1-ubyte", "train-images.idx3-ubyte");
        
        //param names: samples, epocs
        if(args.Length < 3){
            Console.WriteLine("\n\n=================\n Please use xxxxx.exe samples epocs pressision\n=========\n");
            return;
        }
        recognizeDigits("recognizeHandWritedDigits.bin", Int32.Parse(args[0]), Int32.Parse(args[1]), double.Parse(args[2]));

    }
}