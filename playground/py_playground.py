def fn_t(a, *args, **kargs):
    print(a)
    print(args)
    print(kargs["b"])
    if "c" in kargs.keys():
        print(kargs["c"])

fn_t("a", b="b", c="c")