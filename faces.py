"""Draws squares around detected faces in the given image."""

import argparse

# [START import_client_library]
from google.cloud import vision
# [END import_client_library]
from google.cloud.vision import types
from PIL import Image, ImageDraw


# [START def_detect_face]
def detect_face(face_file, max_results=4):
    """Uses the Vision API to detect faces in the given file.
    Args:
        face_file: A file-like object containing an image with faces.
    Returns:
        An array of Face objects with information about the picture.
    """
    # [START get_vision_service]
    client = vision.ImageAnnotatorClient()
    # [END get_vision_service]

    content = face_file.read()
    image = types.Image(content=content)

    return client.face_detection(image=image).face_annotations
# [END def_detect_face]


# [START def_highlight_faces]
def highlight_faces(image, faces, output_filename):
    """Draws a polygon around the faces, then saves to output_filename.
    Args:
      image: a file containing the image with the faces.
      faces: a list of faces found in the file. This should be in the format
          returned by the Vision API.
      output_filename: the name of the image file to be created, where the
          faces have polygons drawn around them.
    """
    UPPER_LIP = 9
    LOWER_LIP = 10
    MOUTH_LEFT = 11
    MOUTH_RIGHT = 12
    #MOUTH_CENTER = 13
    mouthLandmarks = (UPPER_LIP, LOWER_LIP, MOUTH_LEFT, MOUTH_RIGHT)

    im = Image.open(image)
    draw = ImageDraw.Draw(im)

    for face in faces:
        facebox = [(vertex.x, vertex.y)
               for vertex in face.bounding_poly.vertices]
        draw.line(facebox + [facebox[0]], width=5, fill='#00ff00')
        mouthbox = []
        for landmark in face.landmarks:
            if landmark.type in mouthLandmarks:
                mouthbox.append((landmark.position.x, landmark.position.y))
        draw.line(mouthbox +[mouthbox[0]], width=2, fill='#0000ff')
    im.save(output_filename)
# [END def_highlight_faces]


# [START def_main]
def main(input_filename, output_filename, max_results):
    with open(input_filename, 'rb') as image:
        faces = detect_face(image, max_results)
        print('Found {} face{}'.format(
            len(faces), '' if len(faces) == 1 else 's'))

        print('Writing to file {}'.format(output_filename))
        # Reset the file pointer, so we can read the file again
        image.seek(0)
        highlight_faces(image, faces, output_filename)
# [END def_main]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Detects faces in the given image.')
    parser.add_argument(
        'input_image', help='the image you\'d like to detect faces in.')
    parser.add_argument(
        '--out', dest='output', default='out.jpg',
        help='the name of the output file.')
    parser.add_argument(
        '--max-results', dest='max_results', default=4,
        help='the max results of face detection.')
    args = parser.parse_args()

    main(args.input_image, args.output, args.max_results)