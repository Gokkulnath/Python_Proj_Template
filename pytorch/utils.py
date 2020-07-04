def calc_batch_size(img):
    *_,h,w,c = img.shape
    mem_per_img = h * w * c * 8
    print ("Total GPU Memory : {} MB".format(torch.cuda.get_device_properties(0).total_memory/(1024*1024))) # In MB 
    

