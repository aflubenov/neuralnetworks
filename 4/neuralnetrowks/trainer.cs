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



/*
    //we evaluate with a "quadratic" cost function plus a regularization. 
    public static double pseudoTrainSetAndEvaluateCost(double[][] inputs, double[][] desired, NeuralNetwork n, double samplesToLearn, double lambdaRegParam, double totalSamples)
    {
        double acumSum = 0.0;
        double tmp = samplesToLearn * 2.0;
        double[][] results = new double[(Int32)samplesToLearn][];
        double sumSquaredWeights = 0;
        //quadratic cost: (1 / 2*n)*Sum( mod(desired-activation)^2) where n = number of training cases, and the sum is over every training case.
        //regularization part: (lambda / 2*M)*sum(weights^2) where M is the total ammount of samples

        //final formula: quadratic cost + regularization
        sumSquaredWeights =  n.pseudoTrainSetSGD(inputs, desired, lambdaRegParam, totalSamples);
        for(Int32 i = 0; i < (Int32)samplesToLearn; i ++)
            results[i] n.Feedfordward(inputs[i]);

        return CrossEntropy.cost(desired, results);
    }
*/
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
                                Int32 epocs, string fileName, double lambdaRegParam, double totalSamples)
    {
        Int32[][] indexes = new Int32[inputs.Length][];
        Int32 i = 0, l = inputs.Length;
        Int32 iterations = 0;
        double[][] aInputs = new double[l][];
        double[][] aDesired = new double[l][];


        //we set this to suffle indexes
        for(i = 0; i < l; i ++)
            indexes[i] = new Int32[1]{i};

   indexes = shuffle<Int32>(indexes);

        //now we are to train by epocs
        for(iterations = 0; iterations < epocs; iterations ++){
            //suffling inputs
         
            for(i = 0; i < l; i ++){
                aInputs[i] = inputs[indexes[i][0]];
                aDesired[i] = desired[indexes[i][0]];
            }

            //we train it
            n.pseudoTrainSetSGD(aInputs, aDesired, lambdaRegParam, totalSamples); //pseudoTrainSetAndEvaluateCost(aInputs, aDesired, n, l, lambdaRegParam, totalSamples);
            if(Console.KeyAvailable && Console.ReadKey(true).Key == ConsoleKey.Escape)
                return false;

            n.FeedfordwardSet(aInputs, aDesired, l);
            Console.SetCursorPosition(0,l+10);        
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


    private static void recognizeDigits(string fileName, Int32 samples, Int32 epocs, double lambdaRegParam, 
                                        double totalSamples, Int32 saveWhenPressision, Int32 startFromSample)
    {
        NeuralNetwork myNet;
        readMNist digits = new readMNist("train-labels.idx1-ubyte", "train-images.idx3-ubyte");
        
        Int32 samplesLearned = 0;
        Int32 minPressision = saveWhenPressision;
        Int32 tmpMinPressision;
        
        double[][] tryiningData = new double[samples][];
        double[][] desired = new double[samples][];
        Int32 tmp = 0;

        if(fileName.Length == 0 || (!File.Exists(fileName) && !File.Exists(fileName+"bkup")))
            myNet = new NeuralNetwork( 784, 1, 1 );
        else if(File.Exists(fileName+"bkup"))
            myNet = NeuralNetwork.getFromFile(fileName+"bkup"); // new NeuralNetwork( 784, 15, 10 );
        else 
            myNet = NeuralNetwork.getFromFile(fileName);

        Console.Clear();
        bool  wannaStop=false;

        //we advance from samples if it is specified
        digits.advanceTo(startFromSample);

        if(startFromSample != 0)
            samplesLearned = startFromSample / samples;

        //the next logic is: if we are already trained with the set of samples, we get more, 
        writeLog(fileName, String.Format("\"id\",\"set#\",\"epocs\",\"test_matches\",\"test_cases\", \"lambda_Reg_\", \"Cost\""));
        while(!wannaStop && samplesLearned*samples < totalSamples){
            
            
            
            //we get one sample 
            for(Int32 i = 0; i < samples; i ++){
                digits.GiveNextValue(out tryiningData[i], ref tmp);
                //desired[i]= getDigits(tmp); 
                desired[i] = numpy.getArrayPopulated<double>(10,0);
                desired[i][tmp]=1;
            }

            Console.SetCursorPosition(0,samples+15);
            Console.WriteLine("Training sample set #{0}...", samplesLearned+1 );
            
            wannaStop = !pseudoTrainSetByEpocs(tryiningData, desired, myNet, epocs, fileName, lambdaRegParam, totalSamples);
            samplesLearned++;
            myNet.saveToFile(fileName+"bkup");


            tmpMinPressision = testNetwork(myNet, 100);
            if(tmpMinPressision >= minPressision){
                minPressision = tmpMinPressision;
                myNet.saveToFile(fileName);
            }

            writeLog(fileName, String.Format("{0},{1},{2},{3},{4},{5},{6}", samplesLearned*samples, samples,epocs, tmpMinPressision, 
                                            100, lambdaRegParam, Quadratic.cost(desired, tryiningData, myNet)));
        }

    }

    public static Int32 testNetwork(NeuralNetwork myNet, Int32 testCases){
        double[] digit;
        double[] result;
        double max = 0.0;
        Int32 numb = 0;
        readMNist digits = new readMNist("t10k-labels.idx1-ubyte", "t10k-images.idx3-ubyte");
        Int32 tmp = 0;
        Int32 digitsMached = 0;

        for(Int32 j = 0; j < testCases; j++){    
            digits.GiveNextValue(out digit, ref tmp);
            result = myNet.Feedfordward(digit);
            max = -1.0;

            for(Int32 i = 0; i < result.Length; i++)
                if(result[i] > max){
                    max = result[i];
                    numb = i;
                }
                
            if(numb == tmp) digitsMached += 1;
        }

        digits.closeAll();
        
        return digitsMached;
    }

    public static void writeLog(string fileName, string text){
        try{
        using (StreamWriter sw = File.AppendText(fileName+".csv")){
            sw.WriteLine(text);
        }
        }catch(Exception e){
            System.Threading.Thread.CurrentThread.Join(1000);
            writeLog(fileName, text);
        }
    }

  
#endregion

    public static void Main(string[] args)
    {

       // readMNist a = new readMNist("train-labels.idx1-ubyte", "train-images.idx3-ubyte");
        
        //param names: samples, epocs
        if(args.Length < 7){
            Console.WriteLine("\n\n=================\n Please use xxxxx.exe [file To Save] [samples SubSet] [epocs] [lambda Reg.Param] [total Samples] [save when pressisiion] [start from sample] \n=========\n");
            return;
        }
        recognizeDigits(args[0], Int32.Parse(args[1]), Int32.Parse(args[2]), double.Parse(args[3]), 
                    double.Parse(args[4]), Int32.Parse(args[5]), Int32.Parse(args[6]) );

    }
}