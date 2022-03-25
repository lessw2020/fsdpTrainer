# Singleton class using metaclass
# keeps a unique dict for global instances

class Singleton(type):
    instances = {}

    def __call__(pClass, *args, **kwargs):

        if pClass not in pClass.instances:

            instance = super().__call__(*args, **kwargs)
            pClass.instances[pClass] = instance

        return pClass.instances[pClass]
