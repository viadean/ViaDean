# Processing DICOM and TIFF images in MATLAB

Processing DICOM and TIFF images in MATLAB can involve various tasks, such as visualization, image segmentation, or quantitative analysis. MATLAB provides robust toolboxes for medical image processing.

Here’s a guide to process **DICOM** and **TIFF** images in MATLAB:

------

### **1. DICOM Image Processing**

#### **Loading DICOM Images**

```matlab
% Read a DICOM file
dicomFile = 'path_to_dicom_file.dcm';
imageData = dicomread(dicomFile);

% Read metadata
metadata = dicominfo(dicomFile);

% Display image
imshow(imageData, [], 'InitialMagnification', 'fit');
title('DICOM Image');
```

#### **Handling DICOM Directories**

```matlab
% Read all DICOM images in a folder
dicomFolder = 'path_to_dicom_folder';
dicomFiles = dir(fullfile(dicomFolder, '*.dcm'));

% Read and store images in a 3D volume
volume = [];
for i = 1:length(dicomFiles)
    filePath = fullfile(dicomFolder, dicomFiles(i).name);
    slice = dicomread(filePath);
    volume(:, :, i) = slice;
end

% Visualize the 3D volume (e.g., slice by slice)
sliceViewer(volume);
```

#### **Basic Image Processing**

- **Histogram Equalization** for contrast adjustment:

```matlab
adjustedImage = histeq(imageData);
imshow(adjustedImage, []);
title('Histogram Equalized DICOM Image');
```

- **Edge Detection**:

```matlab
edges = edge(imageData, 'Canny');
imshow(edges);
title('Edge Detection on DICOM Image');
```

------

### **2. TIFF Image Processing (e.g., Histology)**

#### **Loading TIFF Images**

```matlab
% Read a single TIFF image
tiffFile = 'path_to_tiff_file.tif';
imageData = imread(tiffFile);

% Display image
imshow(imageData);
title('TIFF Image');
```

#### **Loading Multi-Page TIFF**

```matlab
% Read multi-page TIFF into a 3D matrix
tiffFile = 'path_to_multilayer_tiff.tif';
info = imfinfo(tiffFile);

% Preallocate 3D matrix
imageStack = zeros(info(1).Height, info(1).Width, length(info), 'uint8');

for i = 1:length(info)
    imageStack(:, :, i) = imread(tiffFile, i);
end

% Visualize layers
montage(imageStack, 'Size', [1 length(info)]);
```

#### **Basic Image Processing**

- **Thresholding**:

```matlab
% Convert to grayscale if needed
grayImage = rgb2gray(imageData);

% Apply Otsu thresholding
level = graythresh(grayImage);
binaryImage = imbinarize(grayImage, level);

imshow(binaryImage);
title('Binary Mask from TIFF');
```

- **Segmentation Example**:

```matlab
% Active contour segmentation
mask = zeros(size(grayImage));
mask(100:200, 100:200) = 1; % Initial mask
segmented = activecontour(grayImage, mask, 100);

imshow(segmented);
title('Segmented TIFF Image');
```

------

### **3. Advanced Analysis**

- **Volume Rendering for DICOM**:

```matlab
vol3D = permute(volume, [2, 1, 3]); % Ensure correct orientation
volumeViewer(vol3D);
```

- **Region-Based Analysis** (applicable to both formats):

```matlab
% Find connected regions
stats = regionprops(binaryImage, 'Area', 'Centroid');
```

- **Histogram Analysis**:

```matlab
% For intensity distribution
imhist(imageData);
title('Intensity Histogram');
```

------

### **4. Toolbox Recommendations**

- **Image Processing Toolbox**: For core functions like `imshow`, `imresize`, `imrotate`.
- **Medical Imaging Toolbox**: Specifically for DICOM handling and analysis.
- **Image Segmentation Toolbox**: For segmentation tasks.

------

These approaches are a starting point for processing CT/MRI DICOM files and histology TIFF images in MATLAB. For more advanced tasks, you might use deep learning (e.g., for segmentation), available via MATLAB's **Deep Learning Toolbox**.
