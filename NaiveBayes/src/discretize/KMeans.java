package discretize;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Enumeration;
import java.util.Iterator;
import java.util.LinkedList;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;

public class KMeans {
    
    protected Instances instances;
    protected Attribute discretizingAttribute, classAttribute;
    
    protected double delta;
    protected double similarityDelta;
    protected int numberOfCenters;
    protected ArrayList<Center> centers;
    
    public KMeans(Attribute discretizingAttribute, Instances instances) {
        this.discretizingAttribute = discretizingAttribute;
        this.instances = instances;
        this.delta = 0.01;
        this.similarityDelta = 0.1;
        this.centers = new ArrayList<Center>();
        this.initNumberOfCenters();
    }
    
    public Instances discretize() {
        this.initCenters();
        this.ascendingSortInstancesByDiscretizingAttribute();
        this.distributeInstancesToCenters();
        this.actualizeCentersValues();
        this.runKMeansAlgorithm();
        this.mergeSimilarCenters();
        this.removeEmptyCenters();
        this.actualizeCentersValues();
        this.sortCentersByValue();
        return this.getNewInstancesDatasetWithChangedDiscretizingAttributeValues();
    }
    
    // adds specified number of centers
    private void initCenters() {
        for(int a=0; a<numberOfCenters; ++a) {
            this.centers.add(new Center());
        }
    }
    
    // sorts all instances by discretizing atribute
    private void ascendingSortInstancesByDiscretizingAttribute() {
        this.instances.sort(discretizingAttribute);
    }
    
    // initial distributes all instances to centers
    // earlier center, smaller discretizing attribute value
    protected void distributeInstancesToCenters() {
        int averageNumberOfInstancesInCenter = this.instances.numInstances()/this.numberOfCenters;
        int numberOfInstances = this.instances.numInstances();
        Center currentCenter = null;
        Iterator<Center> centersIterator = this.centers.iterator();
        for(int instanceIndex = 0; instanceIndex < numberOfInstances; ++instanceIndex) {
            if(instanceIndex % averageNumberOfInstancesInCenter == 0 && centersIterator.hasNext()) {
                currentCenter = centersIterator.next();
            }
            currentCenter.putNeighbour(this.instances.instance(instanceIndex));
        }
    }
    
    // calculates average discretizing attribute value in center
    private void actualizeCentersValues() {
        Iterator<Center> centersIterator = this.centers.iterator();
        while(centersIterator.hasNext()) {
            Center currentCenter = centersIterator.next();
            currentCenter.actualizeValue();
        }
    }
    
    // runs KMeans algorithm. It finishes when stop condition is reached - algorithm progress is too low
    private void runKMeansAlgorithm() {
        double previousSums = this.getAllSumsOfNeighboursSquaredLengthToCenter();
        while(true) {
            Iterator<Center> examinedCentersIterator = this.centers.iterator();
            while(examinedCentersIterator.hasNext()) {
                Center currentExaminedCenter = examinedCentersIterator.next();
                Iterator<Center> possibleCenterIterator = this.centers.iterator();
                while(possibleCenterIterator.hasNext()) {
                    Center currentPossibleCenter = possibleCenterIterator.next();
                    if(currentPossibleCenter != currentExaminedCenter) {
                        Iterator<Instance> currentExaminedCenterInstancesIterator = currentExaminedCenter.getIterator();
                        while(currentExaminedCenterInstancesIterator.hasNext()) {
                            Instance currentExaminedInstance = currentExaminedCenterInstancesIterator.next();
                            if(currentPossibleCenter.lengthToCenter(currentExaminedInstance) < 
                               currentExaminedCenter.lengthToCenter(currentExaminedInstance)) {
                                currentExaminedCenterInstancesIterator.remove();
                                currentPossibleCenter.putNeighbour(currentExaminedInstance);
                            }
                        }
                    }
                }
            }
            this.actualizeCentersValues();
            double currentSums = this.getAllSumsOfNeighboursSquaredLengthToCenter();
            if(Math.abs(previousSums-currentSums) < this.delta) {
                break;
            } else {
                previousSums = currentSums;
            }
        }
    }
    
    // merges centers with very similar average discretizing attribute value to reduce number of centers
    private void mergeSimilarCenters() {
        for(int examinedCenterIndex = 0; examinedCenterIndex < this.centers.size(); ++examinedCenterIndex) {
            Center currentExaminedCenter = this.centers.get(examinedCenterIndex);
            Iterator<Center> possibleCenterIterator = this.centers.iterator();
            while(possibleCenterIterator.hasNext()) {
                Center currentPossibleCenter = possibleCenterIterator.next();
                if(currentExaminedCenter != currentPossibleCenter && 
                   Math.abs(currentExaminedCenter.value-currentPossibleCenter.value) < this.similarityDelta) {
                    currentExaminedCenter.takeAllInstancesFromCenter(currentPossibleCenter);
                    possibleCenterIterator.remove();
                }
            }
        }
            
    }
    
    // removes empty centers
    private void removeEmptyCenters() {
        Iterator<Center> centersIterator = this.centers.iterator();
        while(centersIterator.hasNext()) {
            Center currentCenter = centersIterator.next();
            if(currentCenter.nearestNeighbors.size()==0) {
                centersIterator.remove();
            }
        }
    }
    
    // sorts centers by average discretizing attribute value
    private void sortCentersByValue() {
        Collections.sort(centers);
    }
    
    // gets new instances set with changed discretizing attribute values to discretized values
    private Instances getNewInstancesDatasetWithChangedDiscretizingAttributeValues() {
        FastVector vector = new FastVector();
        Iterator<Center> centersIterator = this.centers.iterator();
        while(centersIterator.hasNext()) {
            Center currentCenter = centersIterator.next();
            String centerName = currentCenter.getValue()+"";
            vector.addElement(centerName);
        }
        Attribute discretizedAttribute = new Attribute(this.discretizingAttribute.name(), vector);
        return createNewInstancesDatasetWithChangedValuesOfDiscretizedAttribute(discretizedAttribute);
    }
    
    // creates new instances set with chenged values of discretized attribute
    private Instances createNewInstancesDatasetWithChangedValuesOfDiscretizedAttribute(Attribute newAttribute) {
        int discretizingAttributeIndex = getAttributeIndex(this.discretizingAttribute);
        FastVector newAttributes = new FastVector();
        for(int a=0; a < this.instances.numAttributes(); ++a) {
            if(a!=discretizingAttributeIndex) {
                newAttributes.addElement(this.instances.attribute(a));
            } else {
                newAttributes.addElement(newAttribute);
            }
        }
        Instances newInstancesDataset = new Instances(this.instances.relationName(), newAttributes, this.instances.numInstances());
        newInstancesDataset.setClassIndex(this.instances.classIndex());
        Iterator<Center> centersIterator = this.centers.iterator();
        int numberOfInstanceAttributes = this.instances.numAttributes();
        while(centersIterator.hasNext()) {
            Center currentCenter = centersIterator.next();
            Iterator<Instance> currentCenterInstancesIterator = currentCenter.getIterator();
            while(currentCenterInstancesIterator.hasNext()) {
                Instance currentInstance = currentCenterInstancesIterator.next();
                Instance newInstance = new Instance(numberOfInstanceAttributes);
                newInstance.setDataset(newInstancesDataset);
                for(int a = 0; a < numberOfInstanceAttributes; ++a) {
                    if(a != discretizingAttributeIndex) {
                        if(currentInstance.attribute(a).isNumeric()) {
                            newInstance.setValue(a, currentInstance.value(a));
                        } else if(currentInstance.attribute(a).isNominal()) {
                            newInstance.setValue(a, currentInstance.stringValue(a));
                        }
                    } else {
                        newInstance.setValue(a, currentCenter.getValue()+"");
                    }
                }
                newInstancesDataset.add(newInstance);
            }
        }
        this.classAttribute = newInstancesDataset.classAttribute();
        return newInstancesDataset;
    }
    
    // gets attribute index
    private int getAttributeIndex(Attribute attribute) {
        Enumeration<Attribute> attributesEnumerator = this.instances.enumerateAttributes();
        int currentIndex = 0;
        while(attributesEnumerator.hasMoreElements()) {
            if(attributesEnumerator.nextElement() == attribute) {
                return currentIndex;
            } else {
                ++currentIndex;
            }
        }
        return -1;
    }
    
    // calculates sums of neighbours squared length to its center
    // helpful during checking stop condition
    private double getAllSumsOfNeighboursSquaredLengthToCenter() {
        Iterator<Center> centersIterator = this.centers.iterator();
        double sums = 0.0;
        while(centersIterator.hasNext()) {
            Center currentCenter = centersIterator.next();
            sums += currentCenter.sumOfNeighboursSquaredLengthToCenter();
        }
        return sums;
    }
    
    public void printCenters() {
        Iterator<Center> centersIterator = this.centers.iterator();
        while(centersIterator.hasNext()) {
            Center currentCenter = centersIterator.next();
            currentCenter.printCenter();
        }
    }
    
    // calculates initial number of centers
    public void initNumberOfCenters() {
        this.numberOfCenters = (int)Math.sqrt(instances.numInstances());
    }
    
    public void setNumberOfCenters(int numberOfCenters) {
        this.numberOfCenters = numberOfCenters;
    }
    
    public void setDeltaOfStopConditionValue(double delta) {
        this.delta = delta;
    }
    
    public void setSimiliarityDelta(double similarityDelta) {
        this.similarityDelta = similarityDelta;
    }
    
    public class Center implements Comparable<Center> {
        
        private double value;//, min, max;
        public LinkedList<Instance> nearestNeighbors;
        
        public Center() {
            this.nearestNeighbors = new LinkedList<Instance>();
        }
        
        public Iterator<Instance> getIterator() {
            return nearestNeighbors.iterator();
        }
        
        public void putNeighbour(Instance newNeigbour) {
            this.nearestNeighbors.add(newNeigbour);
        }
        
        public void removeNeighbour(Instance neighbour) {
            this.nearestNeighbors.remove(neighbour);
        }
        
        // calculates length to center
        // helpful during checking if center of neighbours should be changed
        public double lengthToCenter(Instance neighbour) {
            double neighbourValue = 
                    neighbour.value(KMeans.this.discretizingAttribute);
            return Math.abs(value-neighbourValue);
        }
        
        public double getSquaredLengthToCenter(Instance neighbour) {
            double neighbourValue = 
                    neighbour.value(KMeans.this.discretizingAttribute);
            return Math.pow(value-neighbourValue, 2.0);
        }
        
        public double sumOfNeighboursSquaredLengthToCenter() {
            Iterator<Instance> neighboursIterator = this.getIterator();
            double sum = 0.0;
            while(neighboursIterator.hasNext()) {
                sum += this.getSquaredLengthToCenter(neighboursIterator.next());
            }
            return sum;
        }
        
        // takes all instances from parameter center and injects to current center object
        public void takeAllInstancesFromCenter(Center center) {
            Iterator<Instance> takingInstancesIterator = center.getIterator();
            while(takingInstancesIterator.hasNext()) {
                this.nearestNeighbors.add(takingInstancesIterator.next());
            }
        }
        
        // calculates average center value - average value of discretizing attribute in center
        public void actualizeValue() {
            Iterator<Instance> neighboursIterator = this.getIterator();
            double averageValue = 0.0;
            while(neighboursIterator.hasNext()) {
                averageValue += 
                        neighboursIterator.next().value(discretizingAttribute);
            }
            int nearestNeighboursSize = this.nearestNeighbors.size();
            this.value = averageValue/((double)(nearestNeighboursSize < 1 ? 1.0 : nearestNeighboursSize));
        }
       
        public double getValue() {
            return this.value;
        }
        
        public void printCenter() {
            System.out.println("Centrum: "+this.value);
            Iterator<Instance> instancesIterator = this.nearestNeighbors.iterator();
            int number = 1;
            while(instancesIterator.hasNext()) {
                Instance currentInstance = instancesIterator.next();
                System.out.print(number+". "+currentInstance.value(discretizingAttribute)+" ");
                ++number;
            }
            System.out.println();
        }

        @Override
        public int compareTo(Center o) {
            return (int)Math.signum(this.value-o.value);
        }
        
    }
    
}
