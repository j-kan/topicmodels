import os
import shutil

def sample(srcdir, destdir, num):
    for subdir in os.listdir(srcdir):
        src_path  = "%s/%s" % (srcdir, subdir)
        dest_path = "%s/%s" % (destdir, subdir)
        os.makedirs(dest_path)
        count = 0
        for filename in os.listdir(src_path):
            src_file  = "%s/%s" % (src_path, filename)
            dest_file = "%s/%s" % (dest_path, filename)
            shutil.copyfile(src_file, dest_file)
            count += 1
            if count > num:
                break
                
sample('../20news-bydate/20news-bydate-train','data', 10)            