/*===============================================================================
 * Copyright (c) 2010-2012 University of Massachusetts.  All Rights Reserved.
 *
 * Use of the RankLib package is subject to the terms of the software license set 
 * forth in the LICENSE file included with this software, and also available at
 * http://people.cs.umass.edu/~vdang/ranklib_license.html
 *===============================================================================
 */

package ciir.umass.edu.metric;

import ciir.umass.edu.learning.RankList;
import ciir.umass.edu.utilities.RankLibError;
import ciir.umass.edu.utilities.Sorter;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;

/**
 * @author mfayoub
 */
public class PCSNDCGScorer extends PCSDCGScorer {

	protected HashMap<String, Double> idealGains = null;

	public PCSNDCGScorer() {
		super();
		idealGains = new HashMap<>();
	}

	public PCSNDCGScorer(int k) {
		super(k);
		idealGains = new HashMap<>();
	}

	public MetricScorer copy() {
		return new PCSNDCGScorer();
	}

	// Ignore because there are no external judgments for LETOR datasets
	public void loadExternalRelevanceJudgment(String qrelFile) {
		// Queries with external relevance judgment will have their cached ideal gain
		// value overridden
		try (BufferedReader in = new BufferedReader(new InputStreamReader(new FileInputStream(qrelFile)))) {
			String content = "";
			String lastQID = "";
			List<Integer> rel = new ArrayList<Integer>();
			int nQueries = 0;
			while ((content = in.readLine()) != null) {
				content = content.trim();
				if (content.length() == 0)
					continue;
				String[] s = content.split(" ");
				String qid = s[0].trim();
				// String docid = s[2].trim();
				int label = (int) Math.rint(Double.parseDouble(s[3].trim()));
				if (lastQID.compareTo("") != 0 && lastQID.compareTo(qid) != 0) {
					int size = (rel.size() > k) ? k : rel.size();
					int[] r = new int[rel.size()];
					for (int i = 0; i < rel.size(); i++)
						r[i] = rel.get(i);
					double ideal = getIdealDCG(r, size);
					idealGains.put(lastQID, ideal);
					rel.clear();
					nQueries++;
				}
				lastQID = qid;
				rel.add(label);
			}
			if (rel.size() > 0) {
				int size = (rel.size() > k) ? k : rel.size();
				int[] r = new int[rel.size()];
				for (int i = 0; i < rel.size(); i++)
					r[i] = rel.get(i);
				double ideal = getIdealDCG(r, size);
				idealGains.put(lastQID, ideal);
				rel.clear();
				nQueries++;
			}
			System.out.println("Relevance judgment file loaded. [#q=" + nQueries + "]");
		} catch (IOException ex) {
			throw RankLibError.create("Error in NDCGScorer::loadExternalRelevanceJudgment(): ", ex);
		}
	}

	/**
	 * Compute NDCG at k. NDCG(k) = DCG(k) / DCG_{perfect}(k). Note that the
	 * "perfect ranking" must be computed based on the whole list, not just top-k
	 * portion of the list.
	 */
	public double score(RankList rl) {
		// System.out.println("----- In score function " + rl.getID());
		if (rl.size() == 0)
			return 0;

		int size = k;
		if (k > rl.size() || k <= 0)
			size = rl.size();

		int[] rel = getRelevanceLabels(rl);

		double[] sense = getSensitivityScores(rl);
		double score = getCSDCG(rel, sense, size);

		double worstDCG = getMinDCG(rel, sense, size);
		double bestDCG = getMaxDCG(rel, sense, size);
		if (score < worstDCG) {
			return 0;
		} else if (score > bestDCG) {
			return 1;
		} else {
			return (score - worstDCG) / (bestDCG - worstDCG);
		}

		/*
		double sigmoid = 1.0 / (1.0 + Math.pow(Math.E, score * (-0.1)));
		return sigmoid;
		*/
	}

	public double[][] swapChange(RankList rl) {
		int size = (rl.size() > k) ? k : rl.size();

		int[] rel = getRelevanceLabels(rl);
		double[] sense = getSensitivityScores(rl);

		double worstDCG = getMinDCG(rel, sense, size);
		double bestDCG = getMaxDCG(rel, sense, size);

		double[][] changes = new double[rl.size()][];
		for (int i = 0; i < rl.size(); i++) {
			changes[i] = new double[rl.size()];
			Arrays.fill(changes[i], 0);
		}

		for (int i = 0; i < size; i++)
			for (int j = i + 1; j < rl.size(); j++)
				// sense of both i and j will be canceled
				// Math.abs(worstDCG) in the numerator will be canceled too
				changes[j][i] = changes[i][j] = (discount(i) - discount(j)) * (gain(rel[i]) - gain(rel[j]))
						/ (bestDCG - worstDCG);

		return changes;
	}

	public String name() {
		return "PCS_NDCG@" + k;
	}

	private double getIdealDCG(int[] rel, int topK) {
		int[] idx = Sorter.sort(rel, false);
		double dcg = 0;
		for (int i = 0; i < topK; i++)
			dcg += gain(rel[idx[i]]) * discount(i);
		return dcg;
	}

	protected double getMinDCG(int[] rel, double[] sense, int topK) {
		double dcg = 0;
		boolean[] visited = new boolean[rel.length];

		// Put non-relative but sensitive documents at the beginning
		//System.out.println("MINNNNNNNNNNNNNNNNNNNN Phase 1");
		int filledSoFarFromLeft = -1;
		for (int i = 0; i < topK; i++) {
			double min = Double.POSITIVE_INFINITY;
			int minIndex = -1;
			for (int j = 0; j < rel.length; j++) {
				// We need to process those documents which are sensitive but not relative. They
				// have the strongest negative effect.
				if (visited[j] || rel[j] != 0 || sense[j] <= 0.0) { // I mean precisely 0
					continue;
				}
				double curr = gain(rel[j]) * discount(i) - sense[j] * SENSITIVITY_CONSTANT;
				if (curr < min) {
					min = curr;
					minIndex = j;
				}
			}
			if (minIndex != -1) {
				//System.out.print(min + " ");
				dcg += min;
				visited[minIndex] = true;
				filledSoFarFromLeft = i;
			}
		}

		// Put relative but sensitive documents at the end. They should have negative
		// effect only.
		//System.out.println("\nPhase 2");
		int filedSoFarFromRight = topK;
		for (int i = topK - 1; i > filledSoFarFromLeft; i--) {
			double min = 0.0; // to disregard documents with positive effect
			int minIndex = -1;
			for (int j = 0; j < rel.length; j++) {
				// We need to process those documents which are sensitive but not relative. They
				// have the strongest negative effect.
				if (visited[j] || rel[j] == 0 || sense[j] <= 0.0) { // I mean precisely 0
					continue;
				}
				double curr = gain(rel[j]) * discount(i) - sense[j] * SENSITIVITY_CONSTANT;
				if (curr < min) {
					min = curr;
					minIndex = j;
				}
			}
			if (minIndex != -1) {
				//System.out.print(min + " ");
				dcg += min;
				visited[minIndex] = true;
				filedSoFarFromRight = i;
			}
		}

		// Now, all remaining documents have zero or positive effect. We search for
		// minimum effect from left to right.
		//System.out.println("\nPhase 3");
		for (int i = filledSoFarFromLeft + 1; i < filedSoFarFromRight; i++) {
			double min = Double.POSITIVE_INFINITY;
			int minIndex = -1;
			for (int j = 0; j < rel.length; j++) {
				if (visited[j]) {
					continue;
				}
				double curr = gain(rel[j]) * discount(i) - sense[j] * SENSITIVITY_CONSTANT;
				if (curr < min) {
					min = curr;
					minIndex = j;
				}
			}
			if (minIndex != -1) {
				//System.out.print(min + " ");
				dcg += min;
				visited[minIndex] = true;
			}
		}
		return dcg;
	}

	protected double getMaxDCG(int[] rel, double[] sense, int topK) {
		double dcg = 0;
		boolean[] visited = new boolean[rel.length];

		// Put relative but not sensitive documents at the beginning
		//System.out.println("MAXXXXXXXXXXXXXXXXXX Phase 1");
		int filledSoFarFromLeft = -1;
		for (int i = 0; i < topK; i++) {
			double max = Double.NEGATIVE_INFINITY;
			int maxIndex = -1;
			for (int j = 0; j < rel.length; j++) {
				// We need to process those documents which are relative but not sensitive. They
				// have the strongest positive effect.
				if (visited[j] || rel[j] == 0 || sense[j] != 0.0) {
					continue;
				}
				double curr = gain(rel[j]) * discount(i) - sense[j] * SENSITIVITY_CONSTANT;
				if (curr > max) {
					max = curr;
					maxIndex = j;
				}
			}
			if (maxIndex != -1) {
				//System.out.print(max + " ");
				dcg += max;
				visited[maxIndex] = true;
				filledSoFarFromLeft = i;
			}
		}

		// Put not relative and not sensitive documents at the end. They should have
		// zero
		// effect only.
		//System.out.println("\nPhase 2");
		int filedSoFarFromRight = topK;
		for (int i = topK - 1; i > filledSoFarFromLeft; i--) {
			double max = Double.NEGATIVE_INFINITY;
			int maxIndex = -1;
			for (int j = 0; j < rel.length; j++) {
				// We need to process those documents which are sensitive but not relative. They
				// have the strongest negative effect.
				if (visited[j] || rel[j] != 0 || sense[j] != 0.0) {
					continue;
				}
				double curr = gain(rel[j]) * discount(i) - sense[j] * SENSITIVITY_CONSTANT;
				if (curr > max) {
					max = curr;
					maxIndex = j;
				}
			}
			if (maxIndex != -1) {
				//System.out.print(max + " ");
				dcg += max;
				visited[maxIndex] = true;
				filedSoFarFromRight = i;
			}
		}

		// Now, all remaining documents have zero or negative effect. We search for
		// minimum effect from left to right.
		//System.out.println("\nPhase 3");
		for (int i = filledSoFarFromLeft + 1; i < filedSoFarFromRight; i++) {
			double max = Double.NEGATIVE_INFINITY;
			int maxIndex = -1;
			for (int j = 0; j < rel.length; j++) {
				if (visited[j]) {
					continue;
				}
				double curr = gain(rel[j]) * discount(i) - sense[j] * SENSITIVITY_CONSTANT;
				if (curr > max) {
					max = curr;
					maxIndex = j;
				}
			}
			if (maxIndex != -1) {
				//System.out.print(max + " ");
				dcg += max;
				visited[maxIndex] = true;
			}
		}
		return dcg;
	}
}
