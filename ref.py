import errno
import os
import operator
import sys
import hashlib
import yaml
import json
import xml.etree.ElementTree as XML
import xml
from xml.dom import minidom
import jenkins
import re
import pkg_resources
from pprint import pformat
import logging
import copy
import itertools
import fnmatch
import six
from jenkins_jobs.errors import JenkinsJobsException
import jenkins_jobs.local_yaml as local_yaml

logger = logging.getLogger(__name__)
MAGIC_MANAGE_STRING = "<!-- Managed by Jenkins Job Builder -->"


# Python 2.6's minidom toprettyxml produces broken output by adding extraneous
# whitespace around data. This patches the broken implementation with one taken
# from Python > 2.7.3
def writexml(self, writer, indent="", addindent="", newl=""):
    # indent = current indentation
    # addindent = indentation to add to higher levels
    # newl = newline string
    writer.write(indent + "<" + self.tagName)

    attrs = self._get_attributes()
    a_names = attrs.keys()
    a_names.sort()

    for a_name in a_names:
        writer.write(" %s=\"" % a_name)
        minidom._write_data(writer, attrs[a_name].value)
        writer.write("\"")
    if self.childNodes:
        writer.write(">")
        if (len(self.childNodes) == 1 and
                self.childNodes[0].nodeType == minidom.Node.TEXT_NODE):
            self.childNodes[0].writexml(writer, '', '', '')
        else:
            writer.write(newl)
            for node in self.childNodes:
                node.writexml(writer, indent + addindent, addindent, newl)
            writer.write(indent)
        writer.write("</%s>%s" % (self.tagName, newl))
    else:
        writer.write("/>%s" % (newl))

# PyXML xml.__name__ is _xmlplus. Check that if we don't have the default
# system version of the minidom, then patch the writexml method
if sys.version_info[:3] < (2, 7, 3) or xml.__name__ != 'xml':
    minidom.Element.writexml = writexml


def deep_format(obj, paramdict):
    """Apply the paramdict via str.format() to all string objects found within
       the supplied obj. Lists and dicts are traversed recursively."""
    # YAML serialisation was originally used to achieve this, but that places
    # limitations on the values in paramdict - the post-format result must
    # still be valid YAML (so substituting-in a string containing quotes, for
    # example, is problematic).
    if hasattr(obj, 'format'):
        try:
            result = re.match('^{obj:(?P<key>\w+)}$', obj)
            if result is not None:
                ret = paramdict[result.group("key")]
            else:
                ret = obj.format(**paramdict)
        except KeyError as exc:
            missing_key = exc.message
            desc = "%s parameter missing to format %s\nGiven:\n%s" % (
                   missing_key, obj, pformat(paramdict))
            raise JenkinsJobsException(desc)
    elif isinstance(obj, list):
        ret = []
        for item in obj:
            ret.append(deep_format(item, paramdict))
    elif isinstance(obj, dict):
        ret = {}
        for item in obj:
            try:
                ret[item.format(**paramdict)] = \
                    deep_format(obj[item], paramdict)
            except KeyError as exc:
                missing_key = exc.message
                desc = "%s parameter missing to format %s\nGiven:\n%s" % (
                    missing_key, obj, pformat(paramdict))
                raise JenkinsJobsException(desc)
    else:
        ret = obj
    return ret


def matches(what, glob_patterns):
    """
    Checks if the given string, ``what``, matches any of the glob patterns in
    the iterable, ``glob_patterns``

    :arg str what: String that we want to test if it matches a pattern
    :arg iterable glob_patterns: glob patterns to match (list, tuple, set,
    etc.)
    """
    return any(fnmatch.fnmatch(what, glob_pattern)
               for glob_pattern in glob_patterns)


class YamlParser(object):
    def __init__(self, config=None):
        self.data = {}
        self.jobs = []
        self.xml_jobs = []
        self.config = config
        self.registry = ModuleRegistry(self.config)
        self.path = ["."]
        if self.config:
            if config.has_section('job_builder') and \
                    config.has_option('job_builder', 'include_path'):
                self.path = config.get('job_builder',
                                       'include_path').split(':')
        self.keep_desc = self.get_keep_desc()

    def get_keep_desc(self):
        keep_desc = False
        if self.config and self.config.has_section('job_builder') and \
                self.config.has_option('job_builder', 'keep_descriptions'):
            keep_desc = self.config.getboolean('job_builder',
                                               'keep_descriptions')
        return keep_desc

    def parse_fp(self, fp):
        data = local_yaml.load(fp, search_path=self.path)
        if data:
            if not isinstance(data, list):
                raise JenkinsJobsException(
                    "The topmost collection in file '{fname}' must be a list,"
                    " not a {cls}".format(fname=getattr(fp, 'name', fp),
                                          cls=type(data)))
            for item in data:
                cls, dfn = next(iter(item.items()))
                group = self.data.get(cls, {})
                if len(item.items()) > 1:
                    n = None
                    for k, v in item.items():
                        if k == "name":
                            n = v
                            break
                    # Syntax error
                    raise JenkinsJobsException("Syntax error, for item "
                                               "named '{0}'. Missing indent?"
                                               .format(n))
                name = dfn['name']
                if name in group:
                    self._handle_dups("Duplicate entry found: '{0}' is "
                                      "already defined".format(name))
                group[name] = dfn
                self.data[cls] = group

    def parse(self, fn):
        with open(fn) as fp:
            self.parse_fp(fp)

    def _handle_dups(self, message):

        if not (self.config and self.config.has_section('job_builder') and
                self.config.getboolean('job_builder', 'allow_duplicates')):
            logger.error(message)
            raise JenkinsJobsException(message)
        else:
            logger.warn(message)

    def getJob(self, name):
        job = self.data.get('job', {}).get(name, None)
        if not job:
            return job
        return self.applyDefaults(job)

    def getJobGroup(self, name):
        return self.data.get('job-group', {}).get(name, None)

    def getJobTemplate(self, name):
        job = self.data.get('job-template', {}).get(name, None)
        if not job:
            return job
        return self.applyDefaults(job)

    def applyDefaults(self, data):
        whichdefaults = data.get('defaults', 'global')
        defaults = self.data.get('defaults', {}).get(whichdefaults, {})
        if defaults == {} and whichdefaults != 'global':
            raise JenkinsJobsException("Unknown defaults set: '{0}'"
                                       .format(whichdefaults))
        newdata = {}
        newdata.update(defaults)
        newdata.update(data)
        return newdata

    def formatDescription(self, job):
        if self.keep_desc:
            description = job.get("description", None)
        else:
            description = job.get("description", '')
        if description is not None:
            job["description"] = description + \
                self.get_managed_string().lstrip()

    def expandYaml(self, jobs_filter=None):
        changed = True
        while changed:
            changed = False
            for module in self.registry.modules:
                if hasattr(module, 'handle_data'):
                    if module.handle_data(self):
                        changed = True

        for job in self.data.get('job', {}).values():
            if jobs_filter and not matches(job['name'], jobs_filter):
                logger.debug("Ignoring job {0}".format(job['name']))
                continue
            logger.debug("Expanding job '{0}'".format(job['name']))
            job = self.applyDefaults(job)
            self.formatDescription(job)
            self.jobs.append(job)
        for project in self.data.get('project', {}).values():
            if jobs_filter and not matches(project['name'], jobs_filter):
                logger.debug("Ignoring job {0}".format(project['name']))
                continue
            logger.debug("Expanding project '{0}'".format(project['name']))
            # use a set to check for duplicate job references in projects
            seen = set()
            for jobspec in project.get('jobs', []):
                if isinstance(jobspec, dict):
                    # Singleton dict containing dict of job-specific params
                    jobname, jobparams = next(iter(jobspec.items()))
                    if not isinstance(jobparams, dict):
                        jobparams = {}
                else:
                    jobname = jobspec
                    jobparams = {}
                job = self.getJob(jobname)
                if job:
                    # Just naming an existing defined job
                    if jobname in seen:
                        self._handle_dups("Duplicate job '{0}' specified "
                                          "for project '{1}'".format(
                                              jobname, project['name']))
                    seen.add(jobname)
                    continue
                # see if it's a job group
                group = self.getJobGroup(jobname)
                if group:
                    for group_jobspec in group['jobs']:
                        if isinstance(group_jobspec, dict):
                            group_jobname, group_jobparams = \
                                next(iter(group_jobspec.items()))
                            if not isinstance(group_jobparams, dict):
                                group_jobparams = {}
                        else:
                            group_jobname = group_jobspec
                            group_jobparams = {}
                        job = self.getJob(group_jobname)
                        if job:
                            if group_jobname in seen:
                                self._handle_dups(
                                    "Duplicate job '{0}' specified for "
                                    "project '{1}'".format(group_jobname,
                                                           project['name']))
                            seen.add(group_jobname)
                            continue
                        template = self.getJobTemplate(group_jobname)
                        # Allow a group to override parameters set by a project
                        d = {}
                        d.update(project)
                        d.update(jobparams)
                        d.update(group)
                        d.update(group_jobparams)
                        # Except name, since the group's name is not useful
                        d['name'] = project['name']
                        if template:
                            self.expandYamlForTemplateJob(d, template,
                                                          jobs_filter)
                    continue
                # see if it's a template
                template = self.getJobTemplate(jobname)
                if template:
                    d = {}
                    d.update(project)
                    d.update(jobparams)
                    self.expandYamlForTemplateJob(d, template, jobs_filter)
                else:
                    raise JenkinsJobsException("Failed to find suitable "
                                               "template named '{0}'"
                                               .format(jobname))
        # check for duplicate generated jobs
        seen = set()
        # walk the list in reverse so that last definition wins
        for job in self.jobs[::-1]:
            if job['name'] in seen:
                self._handle_dups("Duplicate definitions for job '{0}' "
                                  "specified".format(job['name']))
                self.jobs.remove(job)
            seen.add(job['name'])

    def expandYamlForTemplateJob(self, project, template, jobs_filter=None):
        dimensions = []
        for (k, v) in project.items():
            if type(v) == list and k not in ['jobs']:
                dimensions.append(zip([k] * len(v), v))
        # XXX somewhat hackish to ensure we actually have a single
        # pass through the loop
        if len(dimensions) == 0:
            dimensions = [(("", ""),)]
        checksums = set([])
        for values in itertools.product(*dimensions):
            params = copy.deepcopy(project)
            params = self.applyDefaults(params)

            expanded_values = {}
            for (k, v) in values:
                if isinstance(v, dict):
                    inner_key = next(iter(v))
                    expanded_values[k] = inner_key
                    expanded_values.update(v[inner_key])
                else:
                    expanded_values[k] = v

            params.update(expanded_values)
            expanded = deep_format(template, params)

            # Keep track of the resulting expansions to avoid
            # regenerating the exact same job.  Whenever a project has
            # different values for a parameter and that parameter is not
            # used in the template, we ended up regenerating the exact
            # same job.
            # To achieve that we serialize the expanded template making
            # sure the dict keys are always in the same order. Then we
            # record the checksum in an unordered unique set which let
            # us guarantee a group of parameters will not be added a
            # second time.
            uniq = json.dumps(expanded, sort_keys=True)
            if six.PY3:
                uniq = uniq.encode('utf-8')
            checksum = hashlib.md5(uniq).hexdigest()

            # Lookup the checksum
            if checksum not in checksums:
                # We also want to skip expansion whenever the user did
                # not ask for that job.
                job_name = expanded.get('name')
                if jobs_filter and not matches(job_name, jobs_filter):
                    continue

                self.formatDescription(expanded)
                self.jobs.append(expanded)
                checksums.add(checksum)

    def get_managed_string(self):
        # The \n\n is not hard coded, because they get stripped if the
        # project does not otherwise have a description.
        return "\n\n" + MAGIC_MANAGE_STRING

    def generateXML(self):
        for job in self.jobs:
            self.xml_jobs.append(self.getXMLForJob(job))

    def getXMLForJob(self, data):
        kind = data.get('project-type', 'freestyle')

        for ep in pkg_resources.iter_entry_points(
                group='jenkins_jobs.projects', name=kind):
            Mod = ep.load()
            mod = Mod(self.registry)
            xml = mod.root_xml(data)
            self.gen_xml(xml, data)
            job = XmlJob(xml, data['name'])
            return job

    def gen_xml(self, xml, data):
        for module in self.registry.modules:
            if hasattr(module, 'gen_xml'):
                module.gen_xml(self, xml, data)


class ModuleRegistry(object):
    entry_points_cache = {}

    def __init__(self, config):
        self.modules = []
        self.modules_by_component_type = {}
        self.handlers = {}
        self.global_config = config

        for entrypoint in pkg_resources.iter_entry_points(
                group='jenkins_jobs.modules'):
            Mod = entrypoint.load()
            mod = Mod(self)
            self.modules.append(mod)
            self.modules.sort(key=operator.attrgetter('sequence'))
            if mod.component_type is not None:
                self.modules_by_component_type[mod.component_type] = mod

    def registerHandler(self, category, name, method):
        cat_dict = self.handlers.get(category, {})
        if not cat_dict:
            self.handlers[category] = cat_dict
        cat_dict[name] = method

    def getHandler(self, category, name):
        return self.handlers[category][name]

    def dispatch(self, component_type,
                 parser, xml_parent,
                 component, template_data={}):
        """This is a method that you can call from your implementation of
        Base.gen_xml or component.  It allows modules to define a type
        of component, and benefit from extensibility via Python
        entry points and Jenkins Job Builder :ref:`Macros <macro>`.

        :arg string component_type: the name of the component
          (e.g., `builder`)
        :arg YAMLParser parser: the global YAML Parser
        :arg Element xml_parent: the parent XML element
        :arg dict template_data: values that should be interpolated into
          the component definition

        See :py:class:`jenkins_jobs.modules.base.Base` for how to register
        components of a module.

        See the Publishers module for a simple example of how to use
        this method.
        """

        if component_type not in self.modules_by_component_type:
            raise JenkinsJobsException("Unknown component type: "
                                       "'{0}'.".format(component_type))

        component_list_type = self.modules_by_component_type[component_type] \
            .component_list_type

        if isinstance(component, dict):
            # The component is a singleton dictionary of name: dict(args)
            name, component_data = next(iter(component.items()))
            if template_data:
                # Template data contains values that should be interpolated
                # into the component definition
                s = yaml.dump(component_data, default_flow_style=False)
                s = s.format(**template_data)
                component_data = yaml.load(s)
        else:
            # The component is a simple string name, eg "run-tests"
            name = component
            component_data = {}

        # Look for a component function defined in an entry point
        eps = ModuleRegistry.entry_points_cache.get(component_list_type)
        if eps is None:
            module_eps = list(pkg_resources.iter_entry_points(
                group='jenkins_jobs.{0}'.format(component_list_type)))
            eps = {}
            for module_ep in module_eps:
                if module_ep.name in eps:
                    raise JenkinsJobsException(
                        "Duplicate entry point found for component type: "
                        "'{0}', '{0}',"
                        "name: '{1}'".format(component_type, name))
                eps[module_ep.name] = module_ep

            ModuleRegistry.entry_points_cache[component_list_type] = eps
            logger.debug("Cached entry point group %s = %s",
                         component_list_type, eps)

        if name in eps:
            func = eps[name].load()
            func(parser, xml_parent, component_data)
        else:
            # Otherwise, see if it's defined as a macro
            component = parser.data.get(component_type, {}).get(name)
            if component:
                for b in component[component_list_type]:
                    # Pass component_data in as template data to this function
                    # so that if the macro is invoked with arguments,
                    # the arguments are interpolated into the real defn.
                    self.dispatch(component_type,
                                  parser, xml_parent, b, component_data)
            else:
                raise JenkinsJobsException("Unknown entry point or macro '{0}'"
                                           " for component type: '{1}'.".
                                           format(name, component_type))


class XmlJob(object):
    def __init__(self, xml, name):
        self.xml = xml
        self.name = name

    def md5(self):
        return hashlib.md5(self.output()).hexdigest()

    def output(self):
        out = minidom.parseString(XML.tostring(self.xml, encoding='UTF-8'))
        return out.toprettyxml(indent='  ', encoding='utf-8')


class CacheStorage(object):
    # ensure each instance of the class has a reference to the required
    # modules so that they are available to be used when the destructor
    # is being called since python will not guarantee that it won't have
    # removed global module references during teardown.
    _yaml = yaml
    _logger = logger

    def __init__(self, jenkins_url, flush=False):
        cache_dir = self.get_cache_dir()
        # One cache per remote Jenkins URL:
        host_vary = re.sub('[^A-Za-z0-9\-\~]', '_', jenkins_url)
        self.cachefilename = os.path.join(
            cache_dir, 'cache-host-jobs-' + host_vary + '.yml')
        if flush or not os.path.isfile(self.cachefilename):
            self.data = {}
        else:
            with file(self.cachefilename, 'r') as yfile:
                self.data = yaml.load(yfile)
        logger.debug("Using cache: '{0}'".format(self.cachefilename))

    @staticmethod
    def get_cache_dir():
        home = os.path.expanduser('~')
        if home == '~':
            raise OSError('Could not locate home folder')
        xdg_cache_home = os.environ.get('XDG_CACHE_HOME') or \
            os.path.join(home, '.cache')
        path = os.path.join(xdg_cache_home, 'jenkins_jobs')
        if not os.path.isdir(path):
            os.makedirs(path)
        return path

    def set(self, job, md5):
        self.data[job] = md5

    def is_cached(self, job):
        if job in self.data:
            return True
        return False

    def has_changed(self, job, md5):
        if job in self.data and self.data[job] == md5:
            return False
        return True

    def save(self):
        # check we initialized sufficiently in case called via __del__
        # due to an exception occurring in the __init__
        if getattr(self, 'data', None) is not None:
            try:
                with open(self.cachefilename, 'w') as yfile:
                    self._yaml.dump(self.data, yfile)
            except Exception as e:
                self._logger.error("Failed to write to cache file '%s' on "
                                   "exit: %s" % (self.cachefilename, e))
            else:
                self._logger.info("Cache saved")
                self._logger.debug("Cache written out to '%s'" %
                                   self.cachefilename)

    def __del__(self):
        self.save()


class Jenkins(object):
    def __init__(self, url, user, password):
        self.jenkins = jenkins.Jenkins(url, user, password)

    def update_job(self, job_name, xml):
        if self.is_job(job_name):
            logger.info("Reconfiguring jenkins job {0}".format(job_name))
            self.jenkins.reconfig_job(job_name, xml)
        else:
            logger.info("Creating jenkins job {0}".format(job_name))
            self.jenkins.create_job(job_name, xml)

    def is_job(self, job_name):
        return self.jenkins.job_exists(job_name)

    def get_job_md5(self, job_name):
        xml = self.jenkins.get_job_config(job_name)
        return hashlib.md5(xml).hexdigest()

    def delete_job(self, job_name):
        if self.is_job(job_name):
            logger.info("Deleting jenkins job {0}".format(job_name))
            self.jenkins.delete_job(job_name)

    def get_jobs(self):
        return self.jenkins.get_jobs()

    def is_managed(self, job_name):
        xml = self.jenkins.get_job_config(job_name)
        try:
            out = XML.fromstring(xml)
            description = out.find(".//description").text
            return description.endswith(MAGIC_MANAGE_STRING)
        except (TypeError, AttributeError):
            pass
        return False


class Builder(object):
    def __init__(self, jenkins_url, jenkins_user, jenkins_password,
                 config=None, ignore_cache=False, flush_cache=False):
        self.jenkins = Jenkins(jenkins_url, jenkins_user, jenkins_password)
        self.cache = CacheStorage(jenkins_url, flush=flush_cache)
        self.global_config = config
        self.ignore_cache = ignore_cache

    def load_files(self, fn):
        self.parser = YamlParser(self.global_config)

        # handle deprecated behavior
        if not hasattr(fn, '__iter__'):
            logger.warning(
                'Passing single elements for the `fn` argument in '
                'Builder.load_files is deprecated. Please update your code '
                'to use a list as support for automatic conversion will be '
                'removed in a future version.')
            fn = [fn]

        files_to_process = []
        for path in fn:
            if os.path.isdir(path):
                files_to_process.extend([os.path.join(path, f)
                                         for f in os.listdir(path)
                                         if (f.endswith('.yml')
                                             or f.endswith('.yaml'))])
            else:
                files_to_process.append(path)

        for in_file in files_to_process:
            # use of ask-for-permissions instead of ask-for-forgiveness
            # performs better when low use cases.
            if hasattr(in_file, 'name'):
                fname = in_file.name
            else:
                fname = in_file
            logger.debug("Parsing YAML file {0}".format(fname))
            if hasattr(in_file, 'read'):
                self.parser.parse_fp(in_file)
            else:
                self.parser.parse(in_file)

    def delete_old_managed(self, keep):
        jobs = self.jenkins.get_jobs()
        for job in jobs:
            if job['name'] not in keep and \
                    self.jenkins.is_managed(job['name']):
                logger.info("Removing obsolete jenkins job {0}"
                            .format(job['name']))
                self.delete_job(job['name'])
            else:
                logger.debug("Ignoring unmanaged jenkins job %s",
                             job['name'])

    def delete_job(self, glob_name, fn=None):
        if fn:
            self.load_files(fn)
            self.parser.expandYaml(glob_name)
            jobs = [j['name']
                    for j in self.parser.jobs
                    if matches(j['name'], [glob_name])]
        else:
            jobs = [glob_name]

        if jobs is not None:
            logger.info("Removing jenkins job(s): %s" % ", ".join(jobs))
        for job in jobs:
            self.jenkins.delete_job(job)
            if(self.cache.is_cached(job)):
                self.cache.set(job, '')

    def delete_all_jobs(self):
        jobs = self.jenkins.get_jobs()
        for job in jobs:
            self.delete_job(job['name'])

    def update_job(self, input_fn, names=None, output=None):
        self.load_files(input_fn)
        self.parser.expandYaml(names)
        self.parser.generateXML()

        self.parser.xml_jobs.sort(key=operator.attrgetter('name'))

        for job in self.parser.xml_jobs:
            if names and not matches(job.name, names):
                continue
            if output:
                if hasattr(output, 'write'):
                    # `output` is a file-like object
                    logger.debug("Writing XML to '{0}'".format(output))
                    try:
                        output.write(job.output())
                    except IOError as exc:
                        if exc.errno == errno.EPIPE:
                            # EPIPE could happen if piping output to something
                            # that doesn't read the whole input (e.g.: the UNIX
                            # `head` command)
                            return
                        raise
                    continue

                output_dir = output

                try:
                    os.makedirs(output_dir)
                except OSError:
                    if not os.path.isdir(output_dir):
                        raise

                output_fn = os.path.join(output_dir, job.name)
                logger.debug("Writing XML to '{0}'".format(output_fn))
                f = open(output_fn, 'w')
                f.write(job.output())
                f.close()
                continue
            md5 = job.md5()
            if (self.jenkins.is_job(job.name)
                    and not self.cache.is_cached(job.name)):
                old_md5 = self.jenkins.get_job_md5(job.name)
                self.cache.set(job.name, old_md5)

            if self.cache.has_changed(job.name, md5) or self.ignore_cache:
                self.jenkins.update_job(job.name, job.output())
                self.cache.set(job.name, md5)
            else:
                logger.debug("'{0}' has not changed".format(job.name))
        return self.parser.xml_jobs