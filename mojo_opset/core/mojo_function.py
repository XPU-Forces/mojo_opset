import os

from torch.autograd import Function


class MojoFuncBase(Function):


    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        is_direct_child = MojoFuncBase in cls.__bases__

        if is_direct_child:
            cls._registry = []
        else:
            family_head = None
            for base in cls.mro()[1:]:
                if base is MojoFuncBase:
                    break
                if MojoFuncBase in base.__bases__:
                    family_head = base
                    break

            if family_head:
                # get default_priority defined in registered class
                default_priority = cls.default_priority if hasattr(cls, 'default_priority') else 0

                # get priority from env var, if not set, use default_priority
                env_var_name = f"{cls.__name__}_PRIORITY".upper()
                env_priority = os.getenv(env_var_name)

                priority = int(env_priority) if env_priority is not None else default_priority

                print(f"Register {cls.__name__} as {family_head.__name__} implementation with priority {priority}")

                if priority in [x[0] for x in family_head._registry]:
                    raise ValueError(f"Operator {cls.__name__} priority {priority} has been registered")

                family_head._registry.append((priority, cls))
                family_head._registry.sort(reverse=False, key=lambda x: x[0])
