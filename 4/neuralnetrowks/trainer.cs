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




    //we evaluate with a "quadratic" cost function plus a regularization. 
    public static double pseudoTrainSetAndEvaluateCost(double[][] inputs, double[][] desired, NeuralNetwork n, double samplesToLearn, double lambdaRegParam)
    {
        double acumSum = 0.0;
        double tmp = samplesToLearn * 2.0;
        double[] results = new double[(Int32)samplesToLearn];
        double sumSquaredWeights = 0;
        //quadratic cost: (1 / 2*n)*Sum( mod(desired-activation)^2) where n = number of training cases, and the sum is over every training case.
        //regularization part: (lambda / 2*n)*sum(weights^2)

        //final formula: quadratic cost + regularization
        sumSquaredWeights =  n.pseudoTrainSetSGD(inputs, desired, lambdaRegParam);
        for (Int32 i = 0; i < samplesToLearn; i++)
        {
            results[i] = numpy.absSqr(numpy.add(desired[i], numpy.scalar(n.Feedfordward(inputs[i]), -1.0)));
            acumSum = acumSum + results[i];

        }

        return (1.0/tmp)*acumSum + ((lambdaRegParam/tmp)*sumSquaredWeights);
    }

    private static Int32 _getMaxIndexFromResults(double[] p){
        Int32 i, l = p.Length;
        Int32 index = -1;
        double max = double.MinValue;

        for(i = 0; i < l; i ++)
            if(p[i] > max){
                max = p[i];
                index = i;
            }

        return index;
    }

    public static bool pseudoTrainSetByEpocs(double[][] inputs, double[][] desired, NeuralNetwork n, 
                                Int32 epocs, string fileName, double lambdaRegParam)
    {
        Int32[][] indexes = new Int32[inputs.Length][];
        Int32 i = 0, l = inputs.Length;
        Int32 iterations = 0;
        double[][] aInputs = new double[l][];
        double[][] aDesired = new double[l][];


        //we set this to suffle indexes
        for(i = 0; i < l; i ++)
            indexes[i] = new Int32[1]{i};

        //now we are to train by epocs
        for(iterations = 0; iterations < epocs; iterations ++){
            //suffling inputs
            indexes = shuffle<Int32>(indexes);

            for(i = 0; i < l; i ++){
                aInputs[i] = inputs[indexes[i][0]];
                aDesired[i] = desired[indexes[i][0]];
            }

            //we train it
            pseudoTrainSetAndEvaluateCost(aInputs, aDesired, n, l, lambdaRegParam);
            if(Console.KeyAvailable && Console.ReadKey(true).Key == ConsoleKey.Escape)
                return false;

            n.FeedfordwardSet(aInputs, aDesired, l);
            Console.SetCursorPosition(0,l+3);        
            Console.WriteLine("Training Epoc # {0} of {1})....",iterations, epocs);
                
        }
        n.saveToFile(fileName+"bkup");

        return true;
        
    }

    private static double[] getDigits(Int32 i){
        double[][] aRet = new double[10][];

        aRet[0] = new double[28]{0,1,1,0,
                                 1,0,0,1,
                                 1,0,0,1,
                                 1,0,0,1,
                                 1,0,0,1,
                                 1,0,0,1,
                                 0,1,1,0,
                                 };
                                
        aRet[1] = new double[28]{0,0,1,0,
                                 0,1,1,0,
                                 1,0,1,0,
                                 0,0,1,0,
                                 0,0,1,0,
                                 0,0,1,0,
                                 0,0,1,0,
                                 };

        aRet[2] = new double[28]{0,1,1,1,
                                 1,0,0,1,
                                 0,0,0,1,
                                 0,0,1,0,
                                 0,1,0,0,
                                 1,0,0,0,
                                 1,1,1,1,
                                 };

        aRet[3] = new double[28]{0,1,1,0,
                                 1,0,0,1,
                                 0,0,0,1,
                                 0,1,1,0,
                                 0,0,0,1,
                                 1,0,0,1,
                                 0,1,1,0,
                                 };
        aRet[4] = new double[28]{1,0,0,1,
                                 1,0,0,1,
                                 1,1,1,1,
                                 0,0,0,1,
                                 0,0,0,1,
                                 0,0,0,1,
                                 0,0,0,1,
                                 };

        aRet[5] = new double[28]{1,1,1,1,
                                 1,0,0,0,
                                 1,1,1,0,
                                 0,0,0,1,
                                 0,0,0,1,
                                 1,0,0,1,
                                 0,1,1,0,
                                 };

        aRet[6] = new double[28]{0,1,1,0,
                                 1,0,0,0,
                                 1,0,0,0,
                                 1,1,1,0,
                                 1,0,0,1,
                                 1,0,0,1,
                                 0,1,1,0,
                                 };
        aRet[7] = new double[28]{0,1,1,1,
                                 1,0,0,1,
                                 0,0,1,0,
                                 0,0,1,0,
                                 0,0,1,0,
                                 0,1,0,0,
                                 0,1,0,0,
                                 };

        aRet[8] = new double[28]{0,1,1,0,
                                 1,0,0,1,
                                 1,0,0,1,
                                 0,1,1,0,
                                 1,0,0,1,
                                 1,0,0,1,
                                 0,1,1,0,
                                 };

        aRet[9] = new double[28]{0,1,1,0,
                                 1,0,0,1,
                                 1,0,0,1,
                                 0,1,1,1,
                                 0,0,0,1,
                                 0,0,0,1,
                                 0,1,1,0,
                                 };

        return aRet[i];

    }


    private static void recognizeDigits(string fileName, Int32 samples, Int32 epocs, double lambdaRegParam)
    {
        NeuralNetwork myNet;
        readMNist digits = new readMNist("train-labels.idx1-ubyte", "train-images.idx3-ubyte");
        //Int32 samples = 1000;
        //Int32 epocs = 5;
        Int32 samplesLearned = 0;
        
        double[][] tryiningData = new double[samples][];
        double[][] desired = new double[samples][];
        Int32 tmp = 0;

        if(fileName.Length == 0 || (!File.Exists(fileName) && !File.Exists(fileName+"bkup")))
            myNet = new NeuralNetwork( 784, 128, 64 ,32, 28 );
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
                desired[i]= getDigits(tmp); //numpy.getArrayPopulated<double>(10,0);
                //desired[i][tmp]=1;
            }

            Console.SetCursorPosition(0,samples+10);
            Console.WriteLine("Training sample set #{0}...", samplesLearned );
            
            wannaStop = !pseudoTrainSetByEpocs(tryiningData, desired, myNet, epocs, fileName, lambdaRegParam);

            myNet.saveToFile(fileName+"bkup");
            myNet.saveToFile(fileName);

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
            Console.WriteLine("\n\n=================\n Please use xxxxx.exe samples epocs lambdaRegParam\n=========\n");
            return;
        }
        recognizeDigits("recognizeHandWritedDigits.bin", Int32.Parse(args[0]), Int32.Parse(args[1]), double.Parse(args[2]));

    }
}