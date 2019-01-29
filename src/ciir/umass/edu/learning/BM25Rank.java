package ciir.umass.edu.learning;

public class BM25Rank extends Ranker {
	
	public static int BM25_INDEX = 41;

	@Override
	public void init() {
	}

	@Override
	public void learn() {
	}

	@Override
	public Ranker createNew() {
		return new BM25Rank();
	}

	@Override
	public String toString() {
		return "";
	}

	@Override
	public String model() {
		String output = "## " + name() + "\n";
		return output;
	}

	@Override
	public void loadFromString(String fullText) {
	}

	@Override
	public String name() {
		return "BM25";
	}

	@Override
	public void printParameters() {
	}

	
	@Override
	public double eval(DataPoint p) {
		return p.getFeatureValue(BM25_INDEX);
	}
}
