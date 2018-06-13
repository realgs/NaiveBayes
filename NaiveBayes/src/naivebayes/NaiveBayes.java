package naivebayes;

import discretize.KMeans;
import java.io.File;
import java.util.Collections;
import java.util.Enumeration;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.Random;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.attribute.Discretize;

// classifier class
public class NaiveBayes extends Classifier{
    
    private LinkedList<AttributeConditionalProbability> conditionalProbabilities;
    private Attribute classAttribute;
    private ClassAttributeProbability classAttributeProbability;
    
    // classifier building
    @Override
    public void buildClassifier(Instances i) throws Exception {
        this.conditionalProbabilities = new LinkedList<AttributeConditionalProbability>();
        this.classAttribute = i.classAttribute();
        this.classAttributeProbability = new ClassAttributeProbability(i);
        
        Enumeration<Attribute> attributesEnum = i.enumerateAttributes();
        while(attributesEnum.hasMoreElements()) {
            Attribute attribute = attributesEnum.nextElement();
            if(attribute.isNominal()) {
                AttributeConditionalProbability attributeProbability 
                        = new AttributeConditionalProbability(attribute, i);
                this.conditionalProbabilities.add(attributeProbability);
            }
        }
    }
    
    @Override
    public double classifyInstance(Instance inst) {
        LinkedList<Pair> classificationProbabilities = new LinkedList<Pair>();
        
        Double sumOfProbabilities = 0.0;
        
        Enumeration<String> classAttributesEnum = this.classAttribute.enumerateValues();
        while(classAttributesEnum.hasMoreElements()) {
            String classAttributeValue = classAttributesEnum.nextElement();
            Double classAffiliationProbability = 1.0;
            classAffiliationProbability *= this.classAttributeProbability.getProbability(classAttributeValue);
            Iterator<AttributeConditionalProbability> conditionalProbabilitiesIterator = 
                    this.conditionalProbabilities.iterator();
            while(conditionalProbabilitiesIterator.hasNext()) {
                AttributeConditionalProbability attributeConditionalProbability = 
                        conditionalProbabilitiesIterator.next();
                classAffiliationProbability 
                        *= attributeConditionalProbability.getConditionalProbability(
                        inst.stringValue(attributeConditionalProbability.getAttribute()), 
                        classAttributeValue);
            }
            classificationProbabilities.add(
                new Pair(
                    (double)this.classAttribute.indexOfValue(classAttributeValue),
                    classAffiliationProbability
                )
            );
            sumOfProbabilities += classAffiliationProbability;
        }
        Iterator<Pair> classificationProbabilitiesIterator = classificationProbabilities.iterator();
        while(classificationProbabilitiesIterator.hasNext()) {
            Pair current = classificationProbabilitiesIterator.next();
            current.probabilityValue/=sumOfProbabilities;
        }
        
        Collections.sort(classificationProbabilities);
        return classificationProbabilities.getFirst().classNumber;
    }
    
    // private class keeping class attribute number and calculated probability
    // of afiliation to it. It's useful due to simplicity of sorting such objects
    // because of Comparable interface implementation
    private class Pair implements Comparable<Pair> {
        
        public Double classNumber;
        public Double probabilityValue;
        
        public Pair(Double classNumber, Double probabilityValue) {
            this.classNumber = classNumber;
            this.probabilityValue = probabilityValue;
        }
        
        @Override
        public int compareTo(Pair o) {
            return -(int)Math.signum(probabilityValue-o.probabilityValue);
        }
        
    }
    
    public static void main(String[] args) throws Exception {
        
        // load instances from arff file
        DataSource source = new DataSource("./data/weather.numeric.arff");
        Instances instances = source.getDataSet();
        instances.setClassIndex(instances.numAttributes()-1);
        
        // Discretize by weka supervised algorithm and evaluate model
        Discretize discretize = new Discretize();
        discretize.setInputFormat(instances);
        Instances instancesDiscretizedByWeka = Filter.useFilter(instances, discretize);
        
        ArffSaver arffSaver = new ArffSaver();
        arffSaver.setInstances(instancesDiscretizedByWeka);
        arffSaver.setFile(new File("./data/instancesDiscretizedByWeka.arff"));
        //arffSaver.setDestination(new File("./data/instancesDiscretizedByWeka.arff"));
        arffSaver.writeBatch();
        
        String[] params = new String[4];
        params[0] = "-t";
        params[1] = "./data/instancesDiscretizedByWeka.arff";
        params[2] = "-x";
        params[3] = "10";
        System.out.println(Evaluation.evaluateModel(new NaiveBayes(), params));
        
        // Discretize by ours KMeans algorithm and evaluate model
        Enumeration<Attribute> attributesEnum = instances.enumerateAttributes();
        Instances instancesDiscretizedByKMeans = instances;
        while(attributesEnum.hasMoreElements()) {
            Attribute attribute = attributesEnum.nextElement();
            if(attribute.isNumeric()) {
                KMeans attributeDiscretization = new KMeans(attribute, instancesDiscretizedByKMeans);
                instancesDiscretizedByKMeans = attributeDiscretization.discretize();
            }
        }
        
        arffSaver = new ArffSaver();
        arffSaver.setInstances(instancesDiscretizedByKMeans);
        arffSaver.setFile(new File("./data/instancesDiscretizedByKMeans.arff"));
        //arffSaver.setDestination(new File("./data/instancesDiscretizedByKMeans.arff"));
        arffSaver.writeBatch();
        
        params[0] = "-t";
        params[1] = "./data/instancesDiscretizedByKMeans.arff";
        params[2] = "-x";
        params[3] = "10";
        System.out.println(Evaluation.evaluateModel(new NaiveBayes(), params));
    }
    
}
