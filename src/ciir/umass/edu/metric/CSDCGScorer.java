/*===============================================================================
 * Copyright (c) 2010-2012 University of Massachusetts.  All Rights Reserved.
 *
 * Use of the RankLib package is subject to the terms of the software license set 
 * forth in the LICENSE file included with this software, and also available at
 * http://people.cs.umass.edu/~vdang/ranklib_license.html
 *===============================================================================
 */

package ciir.umass.edu.metric;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Hashtable;

import ciir.umass.edu.learning.RankList;
import ciir.umass.edu.utilities.SimpleMath;

public class CSDCGScorer extends MetricScorer {

	protected static double[] discount = null;// cache
	protected static double[] gain = null;// cache
	// docid --> sensitivity score {0, 1} for now. Can be used for multiple
	// sensitivity score levels
	protected static Hashtable<Integer, Integer> trueLabels;
	public static double SENSITIVITY_CONSTANT = 1.0;

	public CSDCGScorer() {
		this.k = 10;
		// init cache if we haven't already done so
		if (discount == null) {
			discount = new double[5000];
			for (int i = 0; i < discount.length; i++)
				discount[i] = 1.0 / SimpleMath.logBase2(i + 2);
			gain = new double[6];
			for (int i = 0; i < 6; i++)
				gain[i] = (1 << i) - 1;// 2^i-1
		}
		trueLabels = loadSensitivityTrueScores("judged-docs.txt");
	}

	public CSDCGScorer(int k) {
		this.k = k;
		// init cache if we haven't already done so
		if (discount == null) {
			discount = new double[5000];
			for (int i = 0; i < discount.length; i++)
				discount[i] = 1.0 / SimpleMath.logBase2(i + 2);
			gain = new double[6];
			for (int i = 0; i < 6; i++)
				gain[i] = (1 << i) - 1;// 2^i - 1
		}
		trueLabels = loadSensitivityTrueScores("judged-docs.txt");
	}

	private Hashtable<Integer, Integer> loadSensitivityTrueScores(String filename) {
		Hashtable<Integer, Integer> result = new Hashtable<Integer, Integer>();
		try {
			BufferedReader br = new BufferedReader(new FileReader(new File(filename)));
			// skip the header file
			String line = br.readLine();
			line = br.readLine();
			while (line != null) {
				String[] tokens = line.split("\t");
				int docid = Integer.parseInt(tokens[0]);
				int trueLabel = Integer.parseInt(tokens[2]);
				result.put(docid, trueLabel);
				line = br.readLine();
			}
			br.close();
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return result;
	}

	public MetricScorer copy() {
		return new CSDCGScorer();
	}

	/**
	 * Compute CS_DCG at k.
	 */
	public double score(RankList rl) {
		if (rl.size() == 0)
			return 0;

		int size = k;
		if (k > rl.size() || k <= 0)
			size = rl.size();

		int[] rel = getRelevanceLabels(rl);
		double[] sense = getSensitivityScores(rl);
		return getCSDCG(rel, sense, size);
	}

	protected double[] getSensitivityScores(RankList rl) {
		double[] result = new double[rl.size()];
		for (int i = 0; i < rl.size(); i++) {
			int docid = Integer.parseInt(rl.get(i).getDescription().split(" ")[2]);
			result[i] = trueLabels.get(docid);
		}
		return result;
	}

	public double[][] swapChange(RankList rl) {
		int[] rel = getRelevanceLabels(rl);
		double[] sense = getSensitivityScores(rl);
		
		int size = (rl.size() > k) ? k : rl.size();
		double[][] changes = new double[rl.size()][];
		for (int i = 0; i < rl.size(); i++)
			changes[i] = new double[rl.size()];

		// for(int i=0;i<rl.size()-1;i++)//ignore K, compute changes from the entire
		// ranked list
		for (int i = 0; i < size; i++)
			for (int j = i + 1; j < rl.size(); j++)
				// Here sense of both of i and j will be canceled. So I kept sense here even it is not used.
				changes[j][i] = changes[i][j] = (discount(i) - discount(j)) * (gain(rel[i]) - gain(rel[j]));

		return changes;
	}

	public String name() {
		return "CS_DCG@" + k;
	}

	protected double getCSDCG(int[] rel, double[] sense, int topK) {
		double dcg = 0;
		for (int i = 0; i < topK; i++)
			dcg += gain(rel[i]) * discount(i) - sense[i] * SENSITIVITY_CONSTANT;
		return dcg;
	}

	// lazy caching
	protected double discount(int index) {
		if (index < discount.length)
			return discount[index];

		// we need to expand our cache
		int cacheSize = discount.length + 1000;
		while (cacheSize <= index)
			cacheSize += 1000;
		double[] tmp = new double[cacheSize];
		System.arraycopy(discount, 0, tmp, 0, discount.length);
		for (int i = discount.length; i < tmp.length; i++)
			tmp[i] = 1.0 / SimpleMath.logBase2(i + 2);
		discount = tmp;
		return discount[index];
	}

	protected double gain(int rel) {
		if (rel < gain.length)
			return gain[rel];

		// we need to expand our cache
		int cacheSize = gain.length + 10;
		while (cacheSize <= rel)
			cacheSize += 10;
		double[] tmp = new double[cacheSize];
		System.arraycopy(gain, 0, tmp, 0, gain.length);
		for (int i = gain.length; i < tmp.length; i++)
			tmp[i] = (1 << i) - 1;// 2^i - 1
		gain = tmp;
		return gain[rel];
	}
}
