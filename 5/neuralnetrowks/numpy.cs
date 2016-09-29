public static class numpy {
	static Random rand = new Random(); //reuse this if you are generating many

	public static double randGauss(){
		double u1 = rand.NextDouble (); //these are uniform(0,1) random doubles
		double u2 = rand.NextDouble ();
		double randStdNormal = Math.Sqrt (-2.0 * Math.Log (u1)) *
			Math.Sin (2.0 * Math.PI * u2); //random normal(0,1)

		double mean = 0; 
		double stdDev = 1;
		return mean + stdDev * randStdNormal;
	}

    public static T[] getArrayPopulated<T>(Int32 size, T value)
    {
        T[] aRet = new T[size];
        for (Int32 i = 0; i < size; i++)
            aRet[i] = value;

        return aRet;
    }

	static public double[] randn1(int n){
		double[] ret = new double[n];

		for (Int32 i = 0; i < n; i ++) {
			ret [i] = randGauss ();
		}

		return ret;
	}

	static public double[][] randn(int n, int d){
		double[][] ret = new double[n][];

		for (Int32 i = 0; i < n; i ++) {
			ret [i] = randn1 (d);
		}

		return ret;
	}

	static public double dot(double[] a, double[] b){
		Int32 l = a.Length;
		double ret = 0.0;
		for (Int32 i = 0; i < l; i ++) {
			ret += (a [i] * b [i]);
		}

		return ret;
	}

    /**
     * for the record: a[colIndex][colVector]
     *                 b[rowIndex][rowVector]
     *
     *    outRes should be: outRes[rowIndex][rowVector]
     */
    static public double[][] matrixMult_T1(double[][] a, double[][] b, ref double[][] outRes){
        Int32 li = a.Length;
        Int32 lj = b.Length;
        double[][] aTmp = outRes;

        var iterations = Enumerable.Range(0, li*lj);
        var pquery = from num in iterations.AsParallel() select num;

        pquery.ForAll((e) => { Int32 row = e/li, col = e%li;  aTmp[row][col] = numpy.dot(a[col], b[row]);} );

        outRes = aTmp;

        return outRes;
    }

    static public double[][] matrixMult_T1(double[][] a, double[][] b){
        Int32 l = b.Length;
        Int32 la = a.Length;
        double[][] aRet = new double[l][];

        for(Int32 i = 0; i < l; i ++)
            aRet[i] = new double[la];

        numpy.matrixMult_T1(a,b, ref aRet);

        return aRet;
    }

    static public double[][] matrixMult(double[] a, double[] b){
        Int32 lb = b.Length;
        Int32 la = a.Length;
        double[][] altera = new double[la][];
        double[][] alterb = new double[lb][];
        double[][] aRet;

        for(Int32 i = 0; i < la; i ++)
            altera[i] = new double[1]{a[i]};

        for(Int32 i = 0; i < lb; i ++)
            alterb[i] = new double[1]{b[i]};

        
        aRet = numpy.matrixMult_T1(altera,alterb);
        
        return aRet;

    }

    static public double[][] matrixMult(double[][] a, double[][] b){
        double[][] aT = numpy.traspose(a);
        double[][] aRet;
        aRet = numpy.matrixMult_T1(aT, b);
        return aRet;
    }

    static public double[][] matrixMult(double[][] a, double[] b){
        double[][] alterB = new double[1][];
        double[][] aRet;

        alterB[0] = b;

        aRet = matrixMult(a,alterB);
        return aRet;
    }

    static public double[] hadamart(double[] a, double[] b){
        Int32 i, l = a.Length;
        double[] aRet = new double[l];

        for(i = 0; i < l; i ++)
            aRet[i] = a[i]*b[i];

        return aRet;
    }

	static public double[][] traspose(double[][] a){
		double[][] ret = new double[a[0].Length][];

		for (Int32 i = 0; i < a[0].Length; i++) {
			ret [i] = new double[a.Length];
		}

		for (Int32 i = 0; i < a.Length; i++)
			for (Int32 j = 0; j < a[0].Length; j++)
				ret [j] [i] = a [i] [j];

		return ret;
	}

	static public double[][] dot(double [][] a, double [][] b){
		double[][] ret = new double[a.Length][];
		double[][] bt = traspose (b);

		for (Int32 i = 0; i < a.Length; i ++)
			ret [i] = new double[b[0].Length];

		for (Int32 i = 0; i < a.Length; i ++)
			for (Int32 j = 0; j < b[0].Length; j ++)
				ret [i] [j] = dot (a [i], bt [j]);

		return ret;

	}

    static public double[] add(double[] a, double[] b){
        Int32 l = a.Length;
        double[] ret = new double[l];

        for(Int32 i = 0; i < l; i ++)
            ret[i] = a[i]+b[i];

        return ret;
    }
    static public double[][] add(double[][] a, double[][] b){
        Int32 la = a.Length, lb = a[0].Length;
        double[][] aRet = new double[la][];
        

        for(Int32 i = 0; i < la; i ++)
            aRet[i] = new double[lb];

        IEnumerable<Int32> iterations = Enumerable.Range(0, la*lb);
        var pquery = from num in iterations.AsParallel() select num;

        pquery.ForAll((e) => { Int32 row = e/lb, col = e%lb;  aRet[row][col] = a[row][col] + b[row][col];});

        return aRet;
    }

    static public double[] scalar(double[] a, double s){
        Int32 l = a.Length;
        double[] ret = new double[l];

        for(Int32 i = 0; i < l; i ++)
            ret[i] = a[i]*s;

        return ret;
    }

    static public double absSqr(double[] a){
        return dot(a,a);
    }

    static public double[] sqr(double[] a){
        Int32 l = a.Length;
        double[] ret = new double[l];

        for(Int32 i = 0; i < l; i ++)
            ret[i] = a[i]*a[i];

        return ret;
    }
}

