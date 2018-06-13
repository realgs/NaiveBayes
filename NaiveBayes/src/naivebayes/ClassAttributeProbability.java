package naivebayes;

import java.util.Enumeration;
import java.util.HashMap;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

// Class storing class attribute probability needed to calculate 
// probability of affiliation instance to particular class
public class ClassAttributeProbability {
    
    protected Instances instances;
    protected Attribute classAttribute;
    
    // hashmap to store probability of attribute value occurence
    protected HashMap<String, Double> classAttributeProbability;
    
    public ClassAttributeProbability(Instances instances) {
        this.instances = instances;
        this.classAttribute = instances.classAttribute();
        this.classAttributeProbability = new HashMap<String, Double>();
        this.calculateAttributeProbabilities();
    }
    
    public double getProbability(String classAttributeValue){
        return this.classAttributeProbability.get(classAttributeValue);
    }
    
    // calculates class attribute probability - result stored in hasmap
    // way of work is easy - count number of instances for each class value 
    // and divide this value by number of all instances
    private void calculateAttributeProbabilities() {
        Enumeration<Instance> instancesEnum = this.instances.enumerateInstances();
        
        Enumeration<String> classAttributeValueEnum = this.classAttribute.enumerateValues();
        while(classAttributeValueEnum.hasMoreElements()) {
            this.classAttributeProbability.put(classAttributeValueEnum.nextElement(), 0.0);
        }
        
        while(instancesEnum.hasMoreElements()) {
            Instance currentInstance = instancesEnum.nextElement();
            Double currentNumberOfClassAttributeInstances = 
                    this.classAttributeProbability.get(currentInstance.stringValue(this.classAttribute));
            this.classAttributeProbability.put(
                    currentInstance.stringValue(this.classAttribute), 
                    currentNumberOfClassAttributeInstances+1.0);
        }
        
        classAttributeValueEnum = this.classAttribute.enumerateValues();
        while(classAttributeValueEnum.hasMoreElements()) {
            String classAttributeValue = classAttributeValueEnum.nextElement();
            Double currentNumberOfClassAttributeValueInstances 
                    = this.classAttributeProbability.get(classAttributeValue);
            Double probabilityOfClassAttributeValue 
                    = currentNumberOfClassAttributeValueInstances/((double)(instances.numInstances()));
            this.classAttributeProbability.put(classAttributeValue, probabilityOfClassAttributeValue);
        }
    }
  
    
}
