package naivebayes;

import java.util.Enumeration;
import java.util.HashMap;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;


public class AttributeConditionalProbability {
    
    protected Instances instances;
    protected Attribute attribute, classAttribute;
    
    // table to store conditional probability of attribute 
    protected ConditionalProbabilityTable conditionalProbabilityTable;
    
    public AttributeConditionalProbability(Attribute attribute, Instances instances) {
        this.attribute = attribute;
        this.classAttribute = instances.classAttribute();
        this.instances = instances;
        this.conditionalProbabilityTable = 
                new ConditionalProbabilityTable(this.attribute, this.classAttribute);
        this.calculateConditionalProbabilities();
    }
    
    public Attribute getAttribute() {
        return this.attribute;
    }
    
    public double getConditionalProbability(String attributeValue, String classAttributeValue){
        return this.conditionalProbabilityTable.getValue(attributeValue, classAttributeValue);
    }
    
    // calculates conditional probabilities for specified class attribute value and attribute value
    private void calculateConditionalProbabilities() {
        // map keeping number of occurencies of each class attribute value
        HashMap<String, Double> classAttributeOccurencies = new HashMap<String, Double>();
        Enumeration<String> classAttributeValueEnum = this.classAttribute.enumerateValues();
        while(classAttributeValueEnum.hasMoreElements()) {
            classAttributeOccurencies.put(classAttributeValueEnum.nextElement(), 0.0);
        }
        
        // Counting number of attribute value occurencies for specific class
        Enumeration<Instance> instancesEnum = this.instances.enumerateInstances();
        while(instancesEnum.hasMoreElements()) {
            Instance currentInstance = instancesEnum.nextElement();
            String attributeValue = currentInstance.stringValue(this.attribute);
            String classAttributeValue = currentInstance.stringValue(this.classAttribute);
            Double currentNumberOfOccurenciesOfClassAttributeValue = 
                    classAttributeOccurencies.get(classAttributeValue);
            classAttributeOccurencies.put(
                    classAttributeValue, 
                    currentNumberOfOccurenciesOfClassAttributeValue+1.0);
            
            Double currentNumberOfAttributeValueOccurenciesForClassValue = 
                    this.conditionalProbabilityTable.getValue(attributeValue, classAttributeValue);
            this.conditionalProbabilityTable.putValue(
                    currentNumberOfAttributeValueOccurenciesForClassValue+1.0, 
                    attributeValue, 
                    classAttributeValue);
        }
        
        // Calculating conditional probabilities 
        classAttributeValueEnum = this.classAttribute.enumerateValues();
        while(classAttributeValueEnum.hasMoreElements()) {
            String classAttributeValue = classAttributeValueEnum.nextElement();
            Double numberOfOccurenciesOfClassAttributeValue = 
                    classAttributeOccurencies.get(classAttributeValue);
            
            Enumeration<String> attributeValueEnum = this.attribute.enumerateValues();
            while(attributeValueEnum.hasMoreElements()) {
                String attributeValue = attributeValueEnum.nextElement();
                Double numberOfAttributeValueOccurencies =
                        this.conditionalProbabilityTable.getValue(attributeValue, classAttributeValue);
                this.conditionalProbabilityTable.putValue(
                        numberOfAttributeValueOccurencies/numberOfOccurenciesOfClassAttributeValue,
                        attributeValue, 
                        classAttributeValue);
            }
        }
        
    }
    
}
