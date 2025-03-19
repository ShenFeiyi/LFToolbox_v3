# -*- coding:utf-8 -*-

class LFImage:
    """Raw Light Field Image class
    """
    def __init__(self, name, image, local_path):
        """
        Args:
            name (str): Image name
            image (numpy.ndarray): Image
            local_path (str): Local path
        """
        self.name = str(name)
        self.image = image
        self.path = local_path
        self.EIs = None # Elemental images
        """
        Structure of EIs:
        EIs (dict) = {
            #row: {
                #col: image (Width, Height, Color)
                }
            }
        """

    def __repr__(self):
        return self.name + ' @ ' + self.path

    @property
    def elemental_images(self, invM, ApCenters):
        if self.EIs is None:
            self.EIs = self._segment_elemental_images(invM, ApCenters)
        return self.EIs

    def _segment_elemental_images(self, invM, ApCenters):
        """Segment elemental images
        Args:
            invM (numpy.ndarray): 2x2 transformation matrix (rotation & shear)
            ApCenters (list): Aperture centers

        Returns:
            EIs (dict): Elemental images {#row:{#col:image (Width,Height,Color)}}
            anchors (dict): Five points in each EI {#row:{#col:{points}}}
                points = {'center':p0, 'UL':p_upperLeft, 'UR':p_upperRight, 'LR':p_lowerRight, 'LL':p_lowerLeft}
        """
        pass

class ApertureImage(LFImage):
    """Aperture Image class
    """
    def __init__(self, image, local_path):
        """
        Args:
            image (numpy.ndarray): Image
            local_path (str): Local path
        """
        super().__init__('Aperture image', image, local_path)

class DotSampleImage(LFImage):
    """Dot Sample Image class
    """
    def __init__(self, name, image, local_path, posx, posy, posz):
        """
        Args:
            name (str): Image name
            image (numpy.ndarray): Image
            local_path (str): Local path
            posx (float): Position x
            posy (float): Position y
            posz (float): Position z
        """
        super().__init__(name, image, local_path)
        self.posx = posx
        self.posy = posy
        self.posz = posz
        self.featPoints = None
        """
        Structure of featPoints:
        featPoints (dict) = {
            #row: {
                #col: (x,y) in local coordinate (pixels)
                }
            }
        """

    @property
    def feature_points(self):
        if self.featPoints is None:
            self.featPoints = self._detect_circles()
        return self.featPoints

    def _detect_circles(self):
        pass

class STSampleImage(DotSampleImage):
    """ST Sample Image class
    """
    def __init__(self, image, local_path, posx, posy):
        """
        Args:
            image (numpy.ndarray): Image
            local_path (str): Local path
            posx (float): Position x
            posy (float): Position y
        """
        name = 'ST Sample @ ({:.1f}, {:.1f})'.format(posx, posy)
        super().__init__(name, image, local_path, posx, posy, 0)

class ObjSampleImage(DotSampleImage):
    """Object Sample Image class
    """
    def __init__(self, image, local_path, posx, posy, posz):
        """
        Args:
            image (numpy.ndarray): Image
            local_path (str): Local path
            posx (float): Position x
            posy (float): Position y
            posz (float): Position z
        """
        name = 'Object Sample @ ({:.1f}, {:.1f}, {:.1f})'.format(posx, posy, posz)
        super().__init__(name, image, local_path, posx, posy, posz)
