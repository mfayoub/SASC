/*===============================================================================
 * Copyright (c) 2010-2012 University of Massachusetts.  All Rights Reserved.
 *
 * Use of the RankLib package is subject to the terms of the software license set 
 * forth in the LICENSE file included with this software, and also available at
 * http://people.cs.umass.edu/~vdang/ranklib_license.html
 *===============================================================================
 */

package ciir.umass.edu.metric;

import java.util.HashMap;

/**
 * @author vdang
 */
public class MetricScorerFactory {

	private static MetricScorer[] mFactory = new MetricScorer[] { new APScorer(), new NDCGScorer(), new DCGScorer(),
			new PrecisionScorer(), new ReciprocalRankScorer(), new BestAtKScorer(), new ERRScorer(), new CSDCGScorer(),
			new CSNDCGScorer(), new CSZNDCGScorer(), new PCSDCGScorer(), new PCSNDCGScorer(), new PCSZNDCGScorer() };
	private static HashMap<String, MetricScorer> map = new HashMap<String, MetricScorer>();

	public MetricScorerFactory() {
		map.put("MAP", new APScorer());
		map.put("NDCG", new NDCGScorer());
		map.put("DCG", new DCGScorer());
		map.put("P", new PrecisionScorer());
		map.put("RR", new ReciprocalRankScorer());
		map.put("BEST", new BestAtKScorer());
		map.put("ERR", new ERRScorer());
		map.put("CS_DCG", new CSDCGScorer());
		map.put("CS_NDCG", new CSNDCGScorer());
		map.put("CS_ZNDCG", new CSZNDCGScorer());
		map.put("PCS_DCG", new PCSDCGScorer());
		map.put("PCS_NDCG", new PCSNDCGScorer());
		map.put("PCS_ZNDCG", new PCSZNDCGScorer());
	}

	public MetricScorer createScorer(METRIC metric) {
		return mFactory[metric.ordinal() - METRIC.MAP.ordinal()].copy();
	}

	public MetricScorer createScorer(METRIC metric, int k) {
		MetricScorer s = mFactory[metric.ordinal() - METRIC.MAP.ordinal()].copy();
		s.setK(k);
		return s;
	}

	public MetricScorer createScorer(String metric)// e.g.: metric = "NDCG@5"
	{
		int k = -1;
		String m = "";
		MetricScorer s = null;
		if (metric.indexOf("@") != -1) {
			m = metric.substring(0, metric.indexOf("@"));
			k = Integer.parseInt(metric.substring(metric.indexOf("@") + 1));
			s = map.get(m.toUpperCase()).copy();
			s.setK(k);
		} else
			s = map.get(metric.toUpperCase()).copy();
		return s;
	}
}
