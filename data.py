'''data.py
Reads CSV files, stores data, access/filter data by variable name
Jonathan Sosa
CS 251 Data Analysis and Visualization
Spring 2022
'''

#imports
import numpy as np
import csv

class Data:
    def __init__(self, filepath=None, headers=None, data=None, header2col=None):
        
        self.filepah = filepath
        self.header = headers
        self.data = data
        self.header2col = header2col
        self.type = None
        
        if filepath != None :
            
            self.read(filepath)
        
        
        '''Data object constructor

        Parameters:
        -----------
        filepath: str or None. Path to data .csv file
        headers: Python list of strings or None. List of strings that explain the name of each
            column of data.
        data: ndarray or None. shape=(N, M).
            N is the number of data samples (rows) in the dataset and M is the number of variables
            (cols) in the dataset.
            2D numpy array of the datasets values, all formatted as floats.
            NOTE: In Week 1, don't worry working with ndarrays yet. Assume it will be passed in
                  as None for now.
        header2col: Python dictionary or None.
                Maps header (var str name) to column index (int).
                Example: "sepal_length" -> 0

        TODO:
        - Declare/initialize the following instance variables:
            - filepath
            - headers
            - data
            - header2col
            - Any others you find helpful in your implementation
        - If `filepath` isn't None, call the `read` method.
        '''
        

    def read(self, filepath):
        
        #obtains the data and filters it
        file = open(filepath)
        csvreader = csv.reader(file)
        header = next(csvreader)
        type = next(csvreader)
        #print("header:")
        #print(header)
        #print("type:")
        #print(type)

        newHeader = []

        for head in header:
            newHeader.append(head.replace(" ", ""))
                    
        header = newHeader

        nT = []

        for t in type:
            nT.append(t.replace(" ", ""))
            
        type = nT

        noNumeric = []

        for i in range (len(type)):
            if type[i] != "numeric":
                noNumeric.append(i)
                
        #print(noNumeric)

        for i in range(len(noNumeric)):
            del header[noNumeric[-i-1]]
            
        #print(header)

        data = []

        for row in csvreader:
            for i in range(len(noNumeric)):
                del row[noNumeric[-i-1]]
                
            data.append(row)
    
        #print (data)
        #print (data)
        #print (self.header)
        
        self.data = np.array(data,dtype=np.float)
        self.header = header
        
        #creates the dictionary 
        dict = {}
        i=(0)
        for t in self.header:
            dict.update({t:i})
            i+=1
            
        #print (dict)
        self.header2col = dict
        self.dims = len(self.header)
        
        
        '''Read in the .csv file `filepath` in 2D tabular format. Convert to numpy ndarray called
        `self.data` at the end (think of this as 2D array or table).

        Format of `self.data`:
            Rows should correspond to i-th data sample.
            Cols should correspond to j-th variable / feature.

        Parameters:
        -----------
        filepath: str or None. Path to data .csv file

        Returns:
        -----------
        None. (No return value).
            NOTE: In the future, the Returns section will be omitted from docstrings if
            there should be nothing returned

        TODO:
        - Read in the .csv file `filepath` to set `self.data`. Parse the file to only store
        numeric columns of data in a 2D tabular format (ignore non-numeric ones). Make sure
        everything that you add is a float.
        - Represent `self.data` (after parsing your CSV file) as an numpy ndarray. To do this:
            - At the top of this file write: import numpy as np
            - Add this code before this method ends: self.data = np.array(self.data)
        - Be sure to fill in the fields: `self.headers`, `self.data`, `self.header2col`.

        NOTE: You may wish to leverage Python's built-in csv module. Check out the documentation here:
        https://docs.python.org/3/library/csv.html

        NOTE: In any CS251 project, you are welcome to create as many helper methods as you'd like.
        The crucial thing is to make sure that the provided method signatures work as advertised.

        NOTE: You should only use the basic Python library to do your parsing.
        (i.e. no Numpy or imports other than csv).
        Points will be taken off otherwise.

        TIPS:
        - If you're unsure of the data format, open up one of the provided CSV files in a text editor
        or check the project website for some guidelines.
        - Check out the test scripts for the desired outputs.
        '''
        
    def updateData(self,data):
        self.data = data

    def __str__(self):
        '''toString method

        (For those who don't know, __str__ works like toString in Java...In this case, it's what's
        called to determine what gets shown when a `Data` object is printed.)

        Returns:
        -----------
        str. A nicely formatted string representation of the data in this Data object.
            Only show, at most, the 1st 5 rows of data
            See the test code for an example output.
        '''
        print(self.header)
        for row in self.data:
            print (row)
            
    def hed (self):
        return self.header

    def get_headers(self):
        '''Get method for headers

        Returns:
        -----------
        Python list of str.
        '''
        return self.header

    def get_mappings(self):
        '''Get method for mapping between variable name and column index

        Returns:
        -----------
        Python dictionary. str -> int
        '''
        return self.header2col

    def get_num_dims(self):
        '''Get method for number of dimensions in each data sample

        Returns:
        -----------
        int. Number of dimensions in each data sample. Same thing as number of variables.
        '''
        return self.data.ndim

    def get_num_samples(self):
        '''Get method for number of data points (samples) in the dataset

        Returns:
        -----------
        int. Number of data samples in dataset.
        '''
        return (len(self.data))

    def get_sample(self, rowInd):
        '''Gets the data sample at index `rowInd` (the `rowInd`-th sample)

        Returns:
        -----------
        ndarray. shape=(num_vars,) The data sample at index `rowInd`
        '''
        return (self.data.size)

    def get_ind (self,header):
        return (self.header2col[header])

    def get_header_indices(self, headers):
        '''Gets the variable (column) indices of the str variable names in `headers`.

        Parameters:
        -----------
        headers: Python list of str. Header names to take from self.data

        Returns:
        -----------
        Python list of nonnegative ints. shape=len(headers). The indices of the headers in `headers`
            list.
        '''
        
        ind = [] 
       
        for i in headers:
            #print (i)
            #print(self.header2col[i])
            ind.append(self.header2col[i])
            
        return ind
   
    def get_all_data(self):
        '''Gets a copy of the entire dataset

        (Week 2)

        Returns:
        -----------
        ndarray. shape=(num_data_samps, num_vars). A copy of the entire dataset.
            NOTE: This should be a COPY, not the data stored here itself.
            This can be accomplished with numpy's copy function.
        '''
        return (np.copy(self.data))

    def head(self):
        '''Return the 1st five data samples (all variables)

        (Week 2)

        Returns:
        -----------
        ndarray. shape=(5, num_vars). 1st five data samples.
        '''
        if self.data.size > 30:
            count = 0
            row,col = self.data.shape
            first = []
            repetition = 5*col
            
            for r in self.data:
                for c in r:
                    if count < repetition:
                        first.append(c)
                        count += 1
                        
            arr = np.array(first)
            return arr.reshape(5,col)
        else:
            return self.data

    def tail(self):
        '''Return the last five data samples (all variables)

        (Week 2)

        Returns:
        -----------
        ndarray. shape=(5, num_vars). Last five data samples.
        '''
        if self.data.size > 30:
            row,col = self.data.shape
            count = 0
            last = []
            repetition = 5*col

            for i in range (row):
                for b in range (col):
                    if count < repetition:
                        #print (-i-1)
                        #print(-b-1)
                        last.append (self.data[-i-1,-b-1])
                        count += 1
            
            lastArr = []
            for x in range (len(last)):
                lastArr.append(last[-x-1])
                        
            arr = np.array(lastArr)
            return arr.reshape(5,col)
        else:
            return self.data

    def limit_samples(self, start_row, end_row):
        '''Update the data so that this `Data` object only stores samples in the contiguous range:
            `start_row` (inclusive), end_row (exclusive)
        Samples outside the specified range are no longer stored.

        (Week 2)

        '''
        self.data = self.data[start_row:end_row]

    def select_data(self, he, rows=[]):
        '''Return data samples corresponding to the variable names in `headers`.
        If `rows` is empty, return all samples, otherwise return samples at the indices specified
        by the `rows` list.

        (Week 2)

        For example, if self.headers = ['a', 'b', 'c'] and we pass in header = 'b', we return
        column #2 of self.data. If rows is not [] (say =[0, 2, 5]), then we do the same thing,
        but only return rows 0, 2, and 5 of column #2.

        Parameters:
        -----------
            headers: Python list of str. Header names to take from self.data
            rows: Python list of int. Indices of subset of data samples to select.
                Empty list [] means take all rows

        Returns:
        -----------
        ndarray. shape=(num_data_samps, len(headers)) if rows=[]
                 shape=(len(rows), len(headers)) otherwise
            Subset of data from the variables `headers` that have row indices `rows`.

        Hint: For selecting a subset of rows from the data ndarray, check out np.ix_
        '''
        
        ind = self.get_header_indices(he)
        
        self.header = he
        
        dict = {}
        i=(0)
        for t in self.header:
            dict.update({t:i})
            i+=1
            
        self.header2col = dict
        
        nData = []
        
        if rows == []:
            nData = self.data[:,ind]
        else:
            sel = np.ix_(rows,ind)
            nData = self.data[sel]
            
        self.data = nData
        
        return nData
    
                
                