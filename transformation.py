'''transformation.py
Perform projections, translations, rotations, and scaling operations on Numpy ndarray data.
Jonathan Sosa
CS 252 Data Analysis Visualization, Spring 2022
'''
from email import header
import numpy as np
import matplotlib.pyplot as plt
import analysis
import data


class Transformation(analysis.Analysis):

    def __init__(self, orig_dataset, data=None):
        '''Constructor for a Transformation object

        Parameters:
        -----------
        orig_dataset: Data object. shape=(N, num_vars).
            Contains the original dataset (only containing all the numeric variables,
            `num_vars` in total).
        data: Data object (or None). shape=(N, num_proj_vars).
            Contains all the data samples as the original, but ONLY A SUBSET of the variables.
            (`num_proj_vars` in total). `num_proj_vars` <= `num_vars`

        TODO:
        - Pass `data` to the superclass constructor.
        - Create an instance variable for `orig_dataset`.
        '''
        self.origDS = orig_dataset
        self.data = data

    def project(self, headers):
        '''Project the original dataset onto the list of data variables specified by `headers`,
        i.e. select a subset of the variables from the original dataset.
        In other words, your goal is to populate the instance variable `self.data`.

        Parameters:
        -----------
        headers: Python list of str. len(headers) = `num_proj_vars`, usually 1-3 (inclusive), but
            there could be more.
            A list of headers (strings) specifying the feature to be projected onto each axis.
            For example: if headers = ['hi', 'there', 'cs251'], then the data variables
                'hi' becomes the 'x' variable,
                'there' becomes the 'y' variable,
                'cs251' becomes the 'z' variable.
            The length of the list matches the number of dimensions onto which the dataset is
            projected — having 'y' and 'z' variables is optional.

        TODO:
        - Create a new `Data` object that you assign to `self.data` (project data onto the `headers`
        variables). Determine and fill in 'valid' values for all the `Data` constructor
        keyword arguments (except you dont need `filepath` because it is not relevant here).
        '''
        self.data = self.origDS
        self.data.select_data(headers)
            
    def get_data_homogeneous(self):
        '''Helper method to get a version of the projected data array with an added homogeneous
        coordinate. Useful for homogeneous transformations.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars+1). The projected data array with an added 'fake variable'
        column of ones on the right-hand side.
            For example: If we have the data SAMPLE (just one row) in the projected data array:
            [3.3, 5.0, 2.0], this sample would become [3.3, 5.0, 2.0, 1] in the returned array.

        NOTE:
        - Do NOT update self.data with the homogenous coordinate.
        '''
        d = self.data.get_all_data()
        row, col= np.shape(d)
        one = np.ones(row)

        ds = np.c_[d,one]
        
        return ds

    def translation_matrix(self, magnitudes):
        ''' Make an M-dimensional homogeneous transformation matrix for translation,
        where M is the number of features in the projected dataset.

        Parameters:
        -----------
        magnitudes: Python list of float.
            Translate corresponding variables in `headers` (in the projected dataset) by these
            amounts.

        Returns:
        -----------
        ndarray. shape=(num_proj_vars+1, num_proj_vars+1). The transformation matrix.

        NOTE: This method just creates the translation matrix. It does NOT actually PERFORM the
        translation!
        '''
        t = np.eye(len(magnitudes)+1)
        col = len(magnitudes)
        t[:col,col] = magnitudes

        #print(t)

        return t

    def scale_matrix(self, magnitudes):
        '''Make an M-dimensional homogeneous scaling matrix for scaling, where M is the number of
        variables in the projected dataset.

        Parameters:
        -----------
        magnitudes: Python list of float.
            Scale corresponding variables in `headers` (in the projected dataset) by these amounts.

        Returns:
        -----------
        ndarray. shape=(num_proj_vars+1, num_proj_vars+1). The scaling matrix.

        NOTE: This method just creates the scaling matrix. It does NOT actually PERFORM the scaling!
        '''
        scaling = np.eye(len(magnitudes)+1)
        
        for i in range(len(magnitudes)):
            scaling[i,i]=magnitudes[i]
            
        return scaling

    def translate(self, magnitudes):
        '''Translates the variables `headers` in projected dataset in corresponding amounts specified
        by `magnitudes`.

        Parameters:
        -----------
        magnitudes: Python list of float.
            Translate corresponding variables in `headers` (in the projected dataset) by these amounts.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The translated data (with all variables in the projected).
            dataset. NOTE: There should be NO homogenous coordinate!

        TODO:
        - Use matrix multiplication to translate the projected dataset, as advertised above.
        - Update `self.data` with a NEW Data object with the SAME `headers` and `header2col`
        dictionary as the current `self.data`, but DIFFERENT data (set to the data you
        transformed in this method). NOTE: The updated `self.data` SHOULD NOT have a homogenous
        coordinate!
        '''
        homo = self.get_data_homogeneous()
        t = self.translation_matrix(magnitudes)
        
        dsTransformed = (t@homo.T).T
        dsTransformed = np.delete(dsTransformed, -1, axis=1)
        
        self.data.updateData(dsTransformed)
        
        return dsTransformed

    def scale(self, magnitudes):
        '''Scales the variables `headers` in projected dataset in corresponding amounts specified
        by `magnitudes`.

        Parameters:
        -----------
        magnitudes: Python list of float.
            Scale corresponding variables in `headers` (in the projected dataset) by these amounts.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The scaled data (with all variables in the projected).
            dataset. NOTE: There should be NO homogenous coordinate!

        TODO:
        - Use matrix multiplication to scale the projected dataset, as advertised above.
        - Update `self.data` with a NEW Data object with the SAME `headers` and `header2col`
        dictionary as the current `self.data`, but DIFFERENT data (set to the data you
        transformed in this method). NOTE: The updated `self.data` SHOULD NOT have a
        homogenous coordinate!
        '''
        homo = self.get_data_homogeneous()
        s = self.scale_matrix(magnitudes)
        
        dsTransformed = (s@homo.T).T
        dsTransformed = np.delete(dsTransformed, -1, axis=1)
        
        self.data.updateData(dsTransformed)
        
        return dsTransformed

    def transform(self, C):
        '''Transforms the PROJECTED dataset by applying the homogeneous transformation matrix `C`.

        Parameters:
        -----------
        C: ndarray. shape=(num_proj_vars+1, num_proj_vars+1).
            A homogeneous transformation matrix.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The projected dataset after it has been transformed by `C`

        TODO:
        - Use matrix multiplication to apply the compound transformation matix `C` to the projected
        dataset.
        - Update `self.data` with a NEW Data object with the SAME `headers` and `header2col`
        dictionary as the current `self.data`, but DIFFERENT data (set to the data you
        transformed in this method). NOTE: The updated `self.data` SHOULD NOT have a homogenous
        coordinate!
        '''
        homo = self.get_data_homogeneous()
        
        dsTransformed = (C@homo.T).T
        dsTransformed = np.delete(dsTransformed, -1, axis=1)
        
        self.data.updateData(dsTransformed)
        
        return dsTransformed

    def normalize_together(self):
        '''Normalize all variables in the projected dataset together by translating the global minimum
        (across all variables) to zero and scaling the global range (across all variables) to one.

        You should normalize (update) the data stored in `self.data`.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The normalized version of the projected dataset.

        NOTE: Given the goal of this project, for full credit you should implement the normalization
        using matrix multiplications (matrix transformations).
        '''
        min = np.min(self.data.get_all_data())
        max = np.max(self.data.get_all_data())
        
        d = self.data.get_all_data()
        row, col= np.shape(d)
        
        m = []
        
        for i in range(col):
            m.append(-min) 
            
        #print(m)
        ds = self.translate(m)
        
        scale = (1/max-min)
        s = []
        
        for i in range(col):
            s.append(scale) 
        
        ds = self.scale(s)
        
        self.data.updateData(ds)
        
        return ds
        
    def normalize_separately(self):
        '''Normalize each variable separately by translating its local minimum to zero and scaling
        its local range to one.

        You should normalize (update) the data stored in `self.data`.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The normalized version of the projected dataset.

        NOTE: Given the goal of this project, for full credit you should implement the normalization
        using matrix multiplications (matrix transformations).
        '''
        min = np.min(self.data.get_all_data(), axis=0)
        max = np.max(self.data.get_all_data(), axis=0)
        
        ds = self.translate(-min)
        
        scale = (1/max-min)
        ds = self.scale(scale)
        
        self.data.updateData(ds)
        
        return ds 
        
    def rotation_matrix_3d(self, header, degrees):
        '''Make an 3-D homogeneous rotation matrix for rotating the projected data
        about the ONE axis/variable `header`.

        Parameters:
        -----------
        header: str. Specifies the variable about which the projected dataset should be rotated.
        degrees: float. Angle (in degrees) by which the projected dataset should be rotated.

        Returns:
        -----------
        ndarray. shape=(4, 4). The 3D rotation matrix with homogenous coordinate.

        NOTE: This method just creates the rotation matrix. It does NOT actually PERFORM the rotation!
        '''
        ind = self.data.get_ind(header)
        
        degrees = np.radians(degrees)
    
        if ind == 0:
            rot = np.eye(4)
            rot[1, 1] = np.cos(degrees)
            rot[1, 2] = -np.sin(degrees)
            rot[2, 2] = np.cos(degrees)
            rot[2, 1] = np.sin(degrees)
            return rot
        elif ind == 1:
            rot = np.eye(4)
            rot[0, 0] = np.cos(degrees)
            rot[0, 2] = np.sin(degrees)
            rot[2, 2] = np.cos(degrees)
            rot[2, 0] = -np.sin(degrees)
            return rot
        elif ind == 2:
            rot = np.eye(4)
            rot[0, 0] = np.cos(degrees)
            rot[1, 0] = np.sin(degrees)
            rot[1, 1] = np.cos(degrees)
            rot[0, 1] = -np.sin(degrees)
            return rot

    def rotate_3d(self, header, degrees):
        '''Rotates the projected data about the variable `header` by the angle (in degrees)
        `degrees`.

        Parameters:
        -----------
        header: str. Specifies the variable about which the projected dataset should be rotated.
        degrees: float. Angle (in degrees) by which the projected dataset should be rotated.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The rotated data (with all variables in the projected).
            dataset. NOTE: There should be NO homogenous coordinate!

        TODO:
        - Use matrix multiplication to rotate the projected dataset, as advertised above.
        - Update `self.data` with a NEW Data object with the SAME `headers` and `header2col`
        dictionary as the current `self.data`, but DIFFERENT data (set to the data you
        transformed in this method). NOTE: The updated `self.data` SHOULD NOT have a
        homogenous coordinate!
        '''
        rotMat = self.rotation_matrix_3d(header,degrees)
        homo = self.get_data_homogeneous()
        
        dsTransformed = (rotMat@homo.T).T
        dsTransformed = np.delete(dsTransformed, -1, axis=1)
        
        self.data.updateData(dsTransformed)
        
        return dsTransformed

    def scatter3d(self, xlim, ylim, zlim, better_view=False):
        '''Creates a 3D scatter plot to visualize data the x, y, and z axes are drawn, but not ticks

        Axis labels are placed next to the POSITIVE direction of each axis.

        Parameters:
        -----------
        xlim: List or tuple indicating the x axis limits. Format: (low, high)
        ylim: List or tuple indicating the y axis limits. Format: (low, high)
        zlim: List or tuple indicating the z axis limits. Format: (low, high)
        better_view: boolean. Change the view so that the Z axis is coming "out"
        '''
        if len(self.data.get_headers()) != 3:
            print("need 3 headers to make a 3d scatter plot")
            return

        headers = self.data.get_headers()
        xyz = self.data.get_all_data()

        if better_view:
            # by default, matplot lib puts the 3rd axis heading up
            # and the second axis heading back.
            # rotate it so that the second axis is up and the third is forward
            R = np.eye(3)
            R[1, 1] = np.cos(np.pi/2)
            R[1, 2] = -np.sin(np.pi/2)
            R[2, 1] = np.sin(np.pi/2)
            R[2, 2] = np.cos(np.pi/2)
            xyz = (R @ xyz.T).T

        fig = plt.figure()
        ax = plt.axes(projection='3d')

        # Scatter plot of data in 3D
        ax.scatter3D(xyz[:, 0], xyz[:, 1], xyz[:, 2])
        ax.plot(xlim, [0, 0], [0, 0], 'k')
        ax.plot([0, 0], ylim, [0, 0], 'k')
        ax.plot([0, 0], [0, 0], zlim, 'k')
        ax.text(xlim[1], 0, 0, headers[0])

        if better_view:
            ax.text( 0, zlim[0], 0, headers[2])
            ax.text( 0, 0, ylim[1], headers[1])
        else:
            ax.text(0, ylim[1], 0, headers[1])
            ax.text(0, 0, zlim[1], headers[2])

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)
        plt.show()

    def scatter_color(self, ind_var, dep_var, c_var, title=None):
        '''Creates a 2D scatter plot with a color scale representing the 3rd dimension.

        Parameters:
        -----------
        ind_var: str. Header of the variable that will be plotted along the X axis.
        dep_var: Header of the variable that will be plotted along the Y axis.
        c_var: Header of the variable that will be plotted along the color axis.
            NOTE: Use a ColorBrewer color palette (e.g. from the `palettable` library).
        title: str or None. Optional title that will appear at the top of the figure.
        '''
        dataSet = self.data.get_all_data()
        x = self.data.header2col[ind_var]
        y = self.data.header2col[dep_var]
        c = self.data.header2col[c_var]
        
        x = dataSet[:,x]
        y = dataSet[:,y]
        c = dataSet[:,c]
        
        plt.xlabel(ind_var)
        plt.ylabel(dep_var)
        #plt.clabel(c_var)
        
        plt.scatter(x,y,c=c,cmap='RdYlGn_r')
        cbar = plt.colorbar(orientation='vertical',extend='both')
        cbar.set_label(label=c_var,size=15)
        plt.title(title)
        plt.show()

    def heatmap(self, headers=None, title=None, cmap="gray"):
        
        '''Generates a heatmap of the specified variables (defaults to all). Each variable is normalized
        separately and represented as its own row. Each individual is represented as its own column.
        Normalizing each variable separately means that one color axis can be used to represent all
        variables, 0.0 to 1.0.

        Parameters:
        -----------
        headers: Python list of str (or None). (Optional) The variables to include in the heatmap.
            Defaults to all variables if no list provided.
        title: str. (Optional) The figure title. Defaults to an empty string (no title will be displayed).
        cmap: str. The colormap string to apply to the heatmap. Defaults to grayscale
            -- black (0.0) to white (1.0)

        Returns:
        -----------
        fig, ax: references to the figure and axes on which the heatmap has been plotted
        '''

        # Create a doppelganger of this Transformation object so that self.data
        # remains unmodified when heatmap is done
        data_clone = data.Data(headers=self.data.get_headers(),
                               data=self.data.get_all_data(),
                               header2col=self.data.get_mappings())
        dopp = Transformation(self.data, data_clone)
        dopp.normalize_separately()

        fig, ax = plt.subplots()
        if title is not None:
            ax.set_title(title)
        ax.set(xlabel="Individuals")

        # Select features to plot
        if headers is None:
            headers = dopp.data.headers
        m = dopp.data.select_data(headers)

        # Generate heatmap
        hmap = ax.imshow(m.T, aspect="auto", cmap=cmap, interpolation='None')

        # Label the features (rows) along the Y axis
        y_lbl_coords = np.arange(m.shape[1]+1) - 0.5
        ax.set_yticks(y_lbl_coords, minor=True)
        y_lbls = [""] + headers
        ax.set_yticklabels(y_lbls)
        ax.grid(linestyle='none')

        # Create and label the colorbar
        cbar = fig.colorbar(hmap)
        cbar.ax.set_ylabel("Normalized Features")

        return fig, ax
    
    def pair_plot (self):
        
        dataSet = self.data.get_all_data()
        
        fig, axs = plt.subplots(len(self.data.get_headers()), len(self.data.get_headers()),figsize=(12, 12))
        
        for xInd in range(len(self.data.get_headers())):
            x = dataSet[:,xInd]
            xlabel = self.data.get_headers()[xInd]
            
            for yInd in range(len(self.data.get_headers())):
                y = dataSet[:,yInd]
                ylabel = self.data.get_headers()[yInd]
                axs[xInd, yInd].scatter(x, y)
                
            
        return fig,axs
    