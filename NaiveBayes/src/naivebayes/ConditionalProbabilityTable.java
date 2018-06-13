package naivebayes;

import java.util.Enumeration;
import java.util.HashMap;
import weka.core.Attribute;

// stores conditional probability for nominal attribute
public class ConditionalProbabilityTable {
    
    protected double[][] table;
    protected HashMap<String, Integer> tableIndexes;
    protected Attribute attribute, classAttribute;
    /*
     *  |  |c1|c2|..|..
     *  |a1|  |  |  |
     *  |a2|  |  |  |
     *  |a3|  |  |  |
     */
    
    public ConditionalProbabilityTable(Attribute attribute, Attribute classAttribute){
        this.attribute = attribute;
        this.classAttribute = classAttribute;
        initTable(attribute, classAttribute);
        initTableIndexes(attribute, classAttribute);
    }

    public void putValue(double value, String attributeValue, String classAttributeValue) {
        String attributeValueName = this.attribute.name()+"-"+attributeValue;
        String classAttributeValueName = this.classAttribute.name()+"-"+classAttributeValue;
        table[tableIndexes.get(attributeValueName)][tableIndexes.get(classAttributeValueName)] = value;
    }
    
    public double getValue(String attributeValue, String classAttributeValue) {
        String attributeValueName = this.attribute.name()+"-"+attributeValue;
        String classAttributeValueName = this.classAttribute.name()+"-"+classAttributeValue;
        return table[tableIndexes.get(attributeValueName)][tableIndexes.get(classAttributeValueName)];
    }
    
    protected void initTable(Attribute attribute, Attribute classAttribute) {
        this.table = new double[attribute.numValues()][classAttribute.numValues()];
        for(int a=0;a<attribute.numValues();++a) {
            for(int b=0;b<classAttribute.numValues();++b) {
                table[a][b] = 0.0;
            }
        }
    }

    protected void initTableIndexes(Attribute attribute, Attribute classAttribute) {
        this.tableIndexes = new HashMap<String, Integer>();
        fillTableIndexesByAttributeValues(attribute);
        fillTableIndexesByAttributeValues(classAttribute);
    }

    protected void fillTableIndexesByAttributeValues(Attribute attribute) {
        Enumeration<String> attributeValues = attribute.enumerateValues();
        for(int index=0; attributeValues.hasMoreElements(); ++index){
            tableIndexes.put(attribute.name()+"-"+attributeValues.nextElement().toString(), index);
        }
    }
    
}
