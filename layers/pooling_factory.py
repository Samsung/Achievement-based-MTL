from layers.pooling import PPM, FlexPPMSum, SppYolo, ASPP


pooling_dict = {
    'ppm': PPM,
    'fppm': FlexPPMSum,
    'spp': SppYolo,
    'aspp': ASPP
}


def get_pooling_module(pooling_type, in_channel, out_channel, bins=None, image_size=None):
    try:
        return pooling_dict[pooling_type.lower()](in_channel, out_channel, bins, image_size)
    except KeyError:
        print('%s is not yet supported.\nSupported pooling list is %s' % (pooling_type, pooling_dict.keys()))
        exit()
